"""Immutable, content-addressed COLMAP frontends and isolated window copies."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time


def image_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_database(database, names):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid COLMAP cache')
        present = {row[0] for row in db.execute(
            'SELECT name FROM images JOIN keypoints USING(image_id) '
            'JOIN descriptors USING(image_id)')}
        if not set(names) <= present:
            raise ValueError('COLMAP cache lacks requested images/features')
        db.execute('SELECT pair_id FROM two_view_geometries LIMIT 1')


def prepare_feature_cache(images, names, cache_root, run, colmap='colmap'):
    """Publish only after both extraction and matching succeed; never reuse partial work."""
    from .colmap_pipeline import _help_has
    import subprocess
    started = time.perf_counter()
    images = Path(images)
    version = subprocess.run([colmap, '-h'], capture_output=True,
                             text=True, errors='replace', check=True)
    identity = {'schema': 1, 'camera': 'OPENCV', 'features': 4096, 'overlap': 10,
                'colmap': (version.stdout or '') + (version.stderr or ''),
                'images': {name: image_digest(images / name) for name in sorted(names)}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_root) / key
    root.mkdir(parents=True, exist_ok=True)
    database, marker = root / 'database.db', root / 'complete.json'
    hit = False
    if database.is_file() and marker.is_file():
        metadata = json.loads(marker.read_text())
        hit = metadata.get('identity') == identity and metadata.get('sha256') == image_digest(database)
        if hit:
            validate_database(database, names)
    if not hit:
        with tempfile.TemporaryDirectory(dir=root, prefix='build_') as temporary:
            target = Path(temporary) / 'database.db'
            image_list = Path(temporary) / 'images.txt'
            image_list.write_text('\n'.join(sorted(names)) + '\n', encoding='utf-8')
            gpu = '--FeatureExtraction.use_gpu' if _help_has(colmap, 'feature_extractor', '--FeatureExtraction.use_gpu') else '--SiftExtraction.use_gpu'
            matching_gpu = '--FeatureMatching.use_gpu' if _help_has(colmap, 'sequential_matcher', '--FeatureMatching.use_gpu') else '--SiftMatching.use_gpu'
            run([colmap, 'feature_extractor', '--database_path', str(target),
                 '--image_path', str(images), '--image_list_path', str(image_list),
                 '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'OPENCV',
                 '--SiftExtraction.max_num_features', '4096', gpu, '1'])
            run([colmap, 'sequential_matcher', '--database_path', str(target),
                 '--SequentialMatching.overlap', '10', matching_gpu, '1'])
            validate_database(target, names)
            target.replace(database)
            pending = root / 'complete.partial.json'
            pending.write_text(json.dumps({'identity': identity, 'sha256': image_digest(database)}), encoding='utf-8')
            pending.replace(marker)
    return {'database': str(database), 'cache_hit': hit, 'key': key,
            'seconds': time.perf_counter() - started}


def isolate_database(source, destination, names):
    """SQLite backup includes WAL data; retain IDs and only within-window pairs."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('The master database must never be mutated')
    validate_database(source, names)
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as master:
        with closing(sqlite3.connect(destination)) as db:
            master.backup(db)
            db.execute('CREATE TEMP TABLE keep_images (image_id INTEGER PRIMARY KEY)')
            db.executemany('INSERT INTO keep_images SELECT image_id FROM images WHERE name=?',
                           [(name,) for name in names])
            for table in ('matches', 'two_view_geometries'):
                db.execute(f'DELETE FROM {table} WHERE CAST(pair_id / 2147483647 AS INTEGER) '
                           'NOT IN keep_images OR pair_id % 2147483647 NOT IN keep_images')
            tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                columns = {r[1] for r in db.execute(f'PRAGMA table_info({quoted})')}
                if 'image_id' in columns and table != 'images':
                    db.execute(f'DELETE FROM {quoted} WHERE image_id NOT IN keep_images')
            db.execute('DELETE FROM images WHERE image_id NOT IN keep_images')
            db.execute('DELETE FROM cameras WHERE camera_id NOT IN (SELECT camera_id FROM images)')
            db.commit()
    return destination
