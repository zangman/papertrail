from absl import app
from absl import logging
from absl import flags
from google.cloud import storage
from google.cloud import bigquery

import pymupdf4llm
import pathlib
import hashlib
import time
import json
import re
import tempfile
import os

_BUCKET = flags.DEFINE_string('bucket', None, 'GCS bucket name', required=True)
_SOURCE_PREFIX = flags.DEFINE_string('source', None, 'GCS source prefix, Ex: "abhigenai/input/raw"', required=True)
_PROCESSED_PREFIX = flags.DEFINE_string(
  'processed', None, 'GCS destination prefix for processed files, Ex:"abhigenai/input/processed"', required=True)
_DRYRUN = flags.DEFINE_boolean('dryrun', False, 'Whether to just extract the data only')


def strip_md(text: str) -> str:
  """Helper function to remove markdown characters."""
  fixed_text = re.sub(r'[#*]', '', text)
  fixed_text = re.sub(r"<[^>]+>", "", fixed_text)
  return fixed_text.strip()


def get_sections(md: str) -> dict:
  """Identify the sections and keys from markdown."""
  root = {}
  stack = [(0, root)]
  lines = [l.strip() for l in md.splitlines() if l.strip() != '']
  for line in lines:
    if line.startswith('#'):
      level = len(line) - len(line.lstrip('#'))
      title = strip_md(line)
      while stack[-1][0] >= level:
        stack.pop()

      parent = stack[-1][1]

      key = title
      i = 2
      while key in parent:
        key = f'{title} ({i})'
        i += 1

      section = {}
      parent[key] = section
      stack.append((level, section))
    else:
      current = stack[-1][1]
      current.setdefault('_content', '')
      if current['_content']:
        current['_content'] += '\n'
      current['_content'] += strip_md(line)

  return root


def section_word_counts(sections, prefix=''):
  """Count words in each section's own content (excludes subsections)."""
  counts = {}
  for title, body in sections.items():
    path = f'{prefix}/{title}' if prefix else title
    counts[path] = len(body.get('_content', '').split())
    nested = {k: v for k, v in body.items() if k != '_content'}
    if nested:
      counts.update(section_word_counts(nested, path))
  return counts


def sections_to_chunks(sections, doc_name, doc_hash, prefix=''):
  """Flatten sections into RAG chunks: one chunk per section with content."""
  chunks = []
  for title, body in sections.items():
    path = f'{prefix}/{title}' if prefix else title
    content = body.get('_content', '').strip()
    if content:
      chunks.append({
        'text': f'{path}\n{content}',
        'metadata': {
          'doc_name': doc_name,
          'doc_hash': doc_hash,
          'section_path': path,
          'word_count': len(content.split()),
        },
      })
    nested = {k: v for k, v in body.items() if k != '_content'}
    chunks.extend(sections_to_chunks(nested, doc_name, doc_hash, path))
  return chunks


def get_structured_data(file_path):
  doc_name = pathlib.Path(file_path).name
  doc_type = 'pdf'

  with open(file_path, 'rb') as f:
    doc_hash = hashlib.file_digest(f, 'sha256').hexdigest()

  ingestion_timestamp = int(time.time())

  pages = pymupdf4llm.to_markdown(file_path, header=False, footer=False, page_chunks=True)
  md = pymupdf4llm.to_markdown(file_path, header=False, footer=False)

  return {
    'doc_name': doc_name,
    'doc_type': doc_type,
    'doc_hash': doc_hash,
    'ingestion_timestamp': ingestion_timestamp,
    'page_count': len(pages),
    'pages': [strip_md(p['text']).strip() for p in pages],
    'sections': get_sections(md),
  }


def j(obj):
  print(json.dumps(obj, indent=4))


def bq_insert_chunks(client, chunks):
  chunk_rows = []
  for c in chunks:
    chunk_rows.append({
      'chunk_id': hashlib.sha256(f'{c["metadata"]["doc_hash"]}/{c["metadata"]["section_path"]}'.encode()).hexdigest(),
      'doc_hash': c['metadata']['doc_hash'],
      'section_path': c['metadata']['section_path'],
      'text': c['text'],
      'word_count': c['metadata']['word_count'],
    })

  errors = client.insert_rows_json('source.chunks', chunk_rows)
  if errors:
    raise RuntimeError(f'chunk insert failed: {errors}')


def bq_insert_document(client, data, object_path):
  doc_rows = []
  doc_rows.append({
    'doc_hash': data['doc_hash'],
    'doc_name': object_path,
    'doc_type': data['doc_type'],
    'ingestion_timestamp': data['ingestion_timestamp'],
    'page_count': data['page_count'],
  })

  errors = client.insert_rows_json('source.documents', doc_rows)

  if errors:
    raise RuntimeError(f'chunk insert failed: {errors}')


def move_to_processed(storage_client, bucket_name, src_path, dst_prefix):
  source_bucket = storage_client.bucket(bucket_name)
  src_blob = source_bucket.blob(src_path)
  dst_name = f'{dst_prefix}/{pathlib.Path(src_path).name}_{int(time.time())}'
  logging.info(f'destination name: {dst_name}')

  source_bucket.copy_blob(src_blob, source_bucket, dst_name)
  source_bucket.delete_blob(src_path)

  logging.info('moved gs://%s/%s -> gs://%s/%s', bucket_name, src_path, bucket_name, dst_name)
  return dst_name


def doc_hash_exists(bq_client, doc_hash):
  query = 'SELECT 1 from `source.documents` WHERE doc_hash = @doc_hash LIMIT 1'
  job = bq_client.query(
    query,
    job_config=bigquery.QueryJobConfig(
      query_parameters=[bigquery.ScalarQueryParameter('doc_hash', 'STRING', doc_hash)]))
  return next(iter(job), None) is not None


def process_file(storage_client, bq_client, bucket_name, object_name, processed_prefix, is_dry_run):
  blob = storage_client.bucket(bucket_name).blob(object_name)
  base_dir = pathlib.Path('/tmp/gcs')
  local_path = base_dir / object_name
  local_path.parent.mkdir(parents=True, exist_ok=True)
  object_path = f'{bucket_name}/{object_name}'

  blob.download_to_filename(local_path)

  with open(local_path, 'rb') as f:
    doc_hash = hashlib.file_digest(f, 'sha256').hexdigest()

  if doc_hash_exists(bq_client, doc_hash):
    logging.info('doc_hash %s already in BigQuery, skipping', doc_hash)
  else:
    data = get_structured_data(local_path)
    chunks = sections_to_chunks(data['sections'], data['doc_name'], data['doc_hash'])
    if is_dry_run:
      print(j(chunks))

    if not is_dry_run:
      bq_insert_chunks(bq_client, chunks)
      bq_insert_document(bq_client, data, object_path)

  if not is_dry_run:
    move_to_processed(storage_client, _BUCKET.value, object_name, processed_prefix)


def main(argv):
  del argv

  storage_client = storage.Client()
  bq_client = bigquery.Client()

  blobs = storage_client.list_blobs(_BUCKET.value, prefix=_SOURCE_PREFIX.value)
  pdf_blobs = [b for b in blobs if b.name.endswith('.pdf') and b.size > 0]
  logging.info('found %d pdf files under %s', len(pdf_blobs), _SOURCE_PREFIX.value)

  failed = []
  for blob in pdf_blobs:
    try:
      logging.info('Processing %s', blob.name)
      process_file(storage_client, bq_client, _BUCKET.value, blob.name, _PROCESSED_PREFIX.value, _DRYRUN.value)
    except Exception:
      logging.exception('failed process %s', blob.name)
      failed.append(blob.name)

  if failed:
    raise RuntimeError(f'{len(failed)} files failed: {failed}')


if __name__ == '__main__':
  app.run(main)

