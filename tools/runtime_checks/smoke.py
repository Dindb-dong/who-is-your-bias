"""Exercise the actual API through loopback TCP, preserving raw evidence outside Git."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
from pathlib import Path
import platform
import re
import socket
import subprocess
import threading
import time
import wave

from .runtime import digest, local, managed_voice


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def code_receipt() -> dict:
    import bias.api
    root = Path(bias.api.__file__).resolve().parents[2]
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True)
    paths = ['api.py', 'engine.py', 'config.py', 'corpus.py', 'dialogue.py', 'motion.py',
             'query_policy.py', 'snapshots.py', 'bindings.py', 'providers.py', 'grounding.py',
             'retrieval.py', 'jensen_retrieval.py', 'temporal.py']
    return {'repository': str(root), 'head': result.stdout.strip() if result.returncode == 0 else None,
            'modules_sha256': {name: digest(root/'src/bias'/name) for name in paths if (root/'src/bias'/name).is_file()},
            'python': platform.python_version(), 'platform': platform.platform()}


@contextmanager
def server(app):
    """The server and client share the same invocation/network namespace."""
    import uvicorn
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0)); listener.listen(128)
    port = listener.getsockname()[1]
    service = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port,
        loop='asyncio', lifespan='on', log_level='warning', access_log=False))
    errors = []

    def start():
        try: service.run(sockets=[listener])
        except BaseException as exc: errors.append(f'{type(exc).__name__}: {exc}')

    worker = threading.Thread(target=start, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 15
        while not service.started:
            if errors or not worker.is_alive(): raise RuntimeError(f'Actual HTTP server failed: {errors}')
            if time.monotonic() >= deadline: raise RuntimeError('Actual HTTP server readiness timed out')
            time.sleep(.02)
        yield port
    finally:
        service.should_exit = True
        worker.join(timeout=10)
        listener.close()
        if worker.is_alive(): raise RuntimeError('Actual HTTP server did not stop')


class Client:
    def __init__(self, port: int, trace: list):
        self.port = port; self.trace = trace

    def request(self, method: str, path: str, body=None, *, binary=False, timeout=40):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        encoded = json.dumps(body).encode() if body is not None else None
        headers = {'Content-Type': 'application/json'} if encoded is not None else {}
        started = time.perf_counter()
        try:
            connection.request(method, path, encoded, headers)
            response = connection.getresponse(); payload = response.read()
            elapsed = (time.perf_counter()-started)*1000
            data = payload if binary else json.loads(payload)
            record = {'method': method, 'path': path, 'request': body, 'status': response.status,
                'headers': dict(response.getheaders()), 'elapsed_ms': round(elapsed, 3),
                'body_sha256': sha(payload), 'body_bytes': len(payload)}
            if not binary: record['response'] = data
            elif response.status >= 400 and 'application/json' in response.getheader('content-type', ''):
                record['response'] = json.loads(payload)
            self.trace.append(record)
            return response.status, data, record
        finally: connection.close()

    def sse(self, body: dict, *, disconnect_after_meta=False):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=40)
        started = time.perf_counter(); events = []; name = ''; lines = []
        try:
            connection.request('POST', '/api/chat/stream', json.dumps(body).encode(),
                               {'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
            response = connection.getresponse()
            if response.status != 200: raise RuntimeError(f'SSE returned HTTP {response.status}')
            while True:
                line = response.readline()
                if not line: break
                value = line.decode('utf-8').rstrip('\r\n')
                if value.startswith('event:'): name = value[6:].strip()
                elif value.startswith('data:'): lines.append(value[5:].lstrip())
                elif value == '' and lines:
                    events.append({'event': name, 'data': json.loads('\n'.join(lines)),
                                   'elapsed_ms': round((time.perf_counter()-started)*1000, 3)})
                    if disconnect_after_meta and name == 'meta': break
                    if name in {'done', 'error'}: break
                    name = ''; lines = []
            record = {'method': 'POST', 'path': '/api/chat/stream', 'request': body,
                'status': response.status, 'headers': dict(response.getheaders()), 'events': events,
                'client_closed_after_meta': disconnect_after_meta,
                'elapsed_ms': round((time.perf_counter()-started)*1000, 3)}
            self.trace.append(record)
            response.close()
            return events, record
        finally: connection.close()


def run(args, settings, corpus, config: dict, preflight: dict) -> dict:
    output = args.output.resolve(); repo = args.repository.resolve()
    if output.is_relative_to(repo) or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Raw HTTP output contains source excerpts and must be outside the Git repository')
    output.mkdir(parents=True, exist_ok=False); args.output = output
    protocol_path = args.protocol or local(repo, 'reports/jensen-runtime-v1/protocol.json')
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('schema_version') != 'jensen-runtime-smoke-protocol.v1':
        raise ValueError('Unsupported functional smoke protocol')
    trace = []; checks = []
    report = {'schema_version': 'jensen-runtime-smoke.v1', 'started_at': datetime.now(timezone.utc).isoformat(),
        'transport': 'actual loopback TCP HTTP with Uvicorn; no ASGI test transport or simulated model',
        'protocol_sha256': digest(protocol_path), 'preflight': preflight, 'api_code': code_receipt(),
        'condition': 'evidence-preview-with-candidate-voice' if args.with_voice else 'evidence-preview',
        'language_model_generated': False, 'model_latency_measured': False, 'checks': checks, 'cases': [],
        'timing_scope': 'API and evidence delivery; generation_ms is preview assembly, not LLM inference'}

    def check(name: str, condition: bool, **details):
        checks.append({'check': name, 'passed': bool(condition), **details})

    def inspect_turn(case_id: str, turn: dict):
        citations = turn.get('citations', [])
        check(case_id+'.evidence_provider', turn.get('provider') == 'evidence')
        check(case_id+'.no_invalid_citations', not turn.get('grounding', {}).get('invalid_citation_ids'))
        verified = True
        for citation in citations:
            evidence = corpus.evidence.get(citation['evidence_id'])
            source = corpus.sources.get(citation['source_id'])
            verified = verified and bool(evidence and source and evidence.source_id == source.id
                and citation['text'] == evidence.text and citation['url'] == str(source.url)
                and source.split == 'train' and citation['locator'] == evidence.locator.model_dump(mode='json'))
        check(case_id+'.original_citation_binding', verified, citation_count=len(citations))
        check(case_id+'.no_uncontextualized_motion', turn.get('behavior', {}).get('motion') is None)
        if citations:
            check(case_id+'.explicit_preview_label', ('Evidence preview' in turn['text']) or ('자료 미리보기' in turn['text']))
        return {'id': case_id, 'response_sha256': sha(turn['text'].encode()), 'response_characters': len(turn['text']),
            'provider': turn['provider'], 'language': turn['language'], 'grounding': turn['grounding'],
            'context': turn.get('context', {}), 'metrics': turn['metrics'],
            'citations': [{'evidence_id': item['evidence_id'], 'source_id': item['source_id'],
                           'url': item['url'], 'original_text_sha256': sha(item['text'].encode())} for item in citations]}

    try:
        from bias.api import create_app
        with managed_voice(args, config), server(create_app(settings, corpus=corpus)) as bound_port:
            client = Client(bound_port, trace)
            status, health, _ = client.request('GET', '/api/health')
            check('health.evidence_configuration', status == 200 and health['status'] == 'ok'
                and health['provider'] == 'evidence' and not health['capabilities']['llm'])
            check('health.frozen_snapshot', health['persona_snapshot'] == preflight['snapshot'])
            check('health.retrieval', health['retrieval']['backend'] == config['text']['retrieval_backend'])
            report['health'] = {key: health[key] for key in ['provider', 'capabilities', 'providers', 'corpus', 'retrieval', 'configuration_only']}
            status, persona, _ = client.request('GET', '/api/persona')
            check('persona.identity', status == 200 and persona['id'] == 'jensen-huang')
            check('persona.motion_binding', persona['bindings']['motion']['status'] == 'matched'
                and persona['bindings']['motion']['persona_id'] == 'jensen-huang')
            check('persona.voice_binding', persona['capabilities']['speech'] == args.with_voice)
            report['bindings'] = persona['bindings']
            status, sources, _ = client.request('GET', '/api/sources')
            expected_sources = {key: value.model_dump(mode='json') for key, value in corpus.sources.items()}
            check('sources.unchanged_records', status == 200 and {s['id']: s for s in sources['sources']} == expected_sources)
            check('sources.train_only', all(s['split'] == 'train' for s in sources['sources']))
            report['source_ids'] = sorted(expected_sources)
            status, asset, record = client.request('GET', config['avatar']['public_path'], binary=True)
            check('avatar.actual_http_bytes', status == 200 and sha(asset) == config['avatar']['sha256']
                and len(asset) == config['avatar']['bytes'])
            report['served_avatar'] = {key: record[key] for key in ['status', 'body_bytes', 'body_sha256', 'elapsed_ms']}

            status, motion, _ = client.request('GET', '/api/motion/library')
            check('motion.available', status == 200 and motion['available'])
            check('motion.measured_catalog', all(motion[key] == value for key, value in preflight['motion'].items()))
            catalog = json.loads(settings.motion_catalog.read_text())
            representative = {}
            for primitive in catalog['primitives']: representative.setdefault(primitive['clip_id'], primitive['id'])
            clips = {item['id']: item for item in catalog['clips']}; report['motion_clip_checks'] = []
            for clip_id, primitive_id in representative.items():
                path = '/api/motion/clip/'+primitive_id
                denied, _, _ = client.request('GET', path)
                allowed, clip, record = client.request('GET', path+'?preview=true')
                check('motion.preview_gate.'+clip_id, denied == 404 and allowed == 200
                    and clip['review']['status'] == 'geometry_preview'
                    and clip['source']['source_id'] == clips[clip_id]['source_id'])
                report['motion_clip_checks'].append({'clip_id': clip_id, 'primitive_id': primitive_id,
                    'gated_http_status': denied, 'preview_http_status': allowed, 'payload_sha256': record['body_sha256']})

            session = None
            for case in protocol['cases']:
                body = {'message': case['message'], 'language': case['language'], 'mode': 'behavior', 'motion_mode': 'geometry_preview'}
                if case['session'] == 'conversation' and session: body['session_id'] = session
                status, turn, record = client.request('POST', '/api/chat', body)
                check(case['id']+'.http', status == 200)
                if status != 200: continue
                if case['session'] == 'conversation':
                    check(case['id']+'.same_session', session is None or session == turn['session_id'])
                    session = turn['session_id']
                item = inspect_turn(case['id'], turn); item['http_ms'] = record['elapsed_ms']
                report['cases'].append(item); context = turn.get('context', {})
                if 'expected_resolution' in case:
                    check(case['id']+'.resolution', context.get('query_resolution', {}).get('method') == case['expected_resolution'])
                if 'expected_grounding' in case:
                    check(case['id']+'.grounding', turn['grounding']['status'] == case['expected_grounding'] and not turn['citations'])
                if 'expected_temporal' in case:
                    check(case['id']+'.temporal', context.get('temporal', {}).get('mode') == case['expected_temporal'])
                if case.get('expected_limitation'): check(case['id']+'.limitation', bool(context.get('evidence_limitation')))
            check('citations.actual_source_inspection', any(item['citations'] for item in report['cases']))

            status, context, _ = client.request('GET', f'/api/sessions/{session}/context')
            rows = context.get('turns', [])
            check('session.four_turns', status == 200 and context.get('turn_count') == 4)
            check('session.provenance', len(rows) == 4 and rows[2]['user_correction']
                and rows[1]['supporting_user_turn_ids'] == [rows[0]['turn_id']]
                and not rows[3]['supporting_user_turn_ids'] and all(not row['claims_semantically_verified'] for row in rows))
            report['session_context'] = context
            status, deleted, _ = client.request('DELETE', f'/api/sessions/{session}')
            absent, _, _ = client.request('GET', f'/api/sessions/{session}/context')
            check('session.delete', status == 200 and deleted.get('deleted') and absent == 404)

            stream_body = {**protocol['stream_case'], 'mode': 'behavior', 'motion_mode': 'geometry_preview'}
            events, stream_record = client.sse(stream_body)
            names = [event['event'] for event in events]
            results = [event['data']['turn'] for event in events if event['event'] == 'result']
            tokens = ''.join(event['data']['text'] for event in events if event['event'] == 'token')
            check('sse.lifecycle', names[0] == 'meta' and names[-1] == 'done' and 'error' not in names
                and len(results) == 1 and 'token' in names)
            if results:
                # Streaming exposes citation markers; the final display text removes them.
                # Compare all non-whitespace content after that explicit transformation,
                # and independently require the same evidence identifiers.
                markers = re.findall(r'\[source:([A-Za-z0-9_.-]+)\]', tokens)
                clean_tokens = re.sub(r'\[source:[A-Za-z0-9_.-]+\]', '', tokens)
                check('sse.display_content_matches_result', ' '.join(clean_tokens.split()) == ' '.join(results[0]['text'].split()))
                check('sse.marker_identity', set(markers) == {item['evidence_id'] for item in results[0]['citations']})
                check('sse.identity', events[0]['data']['session_id'] == results[0]['session_id'])
                report['stream_turn'] = inspect_turn('sse', results[0])
            report['sse'] = {'events': names, 'first_event_ms': events[0]['elapsed_ms'],
                'first_text_ms': next((event['elapsed_ms'] for event in events if event['event'] == 'token'), None),
                'complete_ms': stream_record['elapsed_ms'], 'raw_token_text_sha256': sha(tokens.encode()),
                'display_transformation': 'remove explicit source markers; normalize whitespace for content comparison',
                'timing_scope': report['timing_scope']}

            events, _ = client.sse(stream_body, disconnect_after_meta=True)
            interrupted_session = events[0]['data']['session_id']
            # A fast preview can finish before the peer close is observed. Record that race.
            time.sleep(.2)
            status, interrupted, _ = client.request('GET', f'/api/sessions/{interrupted_session}/context')
            deleted_status, _, _ = client.request('DELETE', f'/api/sessions/{interrupted_session}')
            check('disconnect.session_unlocked', status == 200 and interrupted.get('turn_count') in {0, 1} and deleted_status == 200)
            report['disconnect'] = {'closed_after_event': 'meta', 'stored_turns': interrupted.get('turn_count'),
                'delete_http_status': deleted_status, 'neural_generation_cancellation_tested': False,
                'scope': 'actual peer disconnect and session release; evidence preview can complete before disconnect'}

            if args.with_voice:
                probe = protocol['speech_probe']
                status, audio, record = client.request('POST', '/api/speech',
                    {'text': probe['text'], 'language': probe['language']}, binary=True, timeout=180)
                check('speech.proxy_http', status == 200)
                if status != 200: raise RuntimeError(f'Actual speech gateway returned HTTP {status}')
                (output/'integration-speech.wav').write_bytes(audio)
                with wave.open(io.BytesIO(audio), 'rb') as wav:
                    frames = wav.getnframes(); rate = wav.getframerate()
                    audio_info = {'sample_rate': rate, 'frames': frames, 'channels': wav.getnchannels(),
                                  'seconds': frames/rate, 'sample_width': wav.getsampwidth()}
                check('speech.nonempty_waveform', frames > 0 and rate > 0)
                headers = {key.lower(): value for key, value in record['headers'].items()}
                check('speech.configured_provider_header', headers.get('x-voice-mode') == 'configured-provider')
                voice_client = Client(args.voice_port, trace)
                status, voice_health, _ = voice_client.request('GET', '/health')
                check('speech.actual_adapted_identity', status == 200 and voice_health.get('persona_id') == 'jensen-huang'
                    and voice_health.get('model_loaded') and (voice_health.get('adaptation') or {}).get('optimizer_steps') == 16)
                report['speech'] = {'text': probe['text'], 'language': probe['language'], 'quality_evaluation': False,
                    'audio_sha256': sha(audio), 'audio_bytes': len(audio), 'http_ms': record['elapsed_ms'],
                    **audio_info, 'provider_health': voice_health, 'headers': record['headers'],
                    'simultaneous_neural_text_model': False}
            status, _, _ = client.request('GET', '/api/health')
            check('health.after_requests', status == 200)
    except Exception as exc:
        report['fatal_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        check('run.completed', False)
    finally:
        write(output/'http-trace.json', trace)
        report['raw_trace'] = {'path': 'http-trace.json', 'sha256': digest(output/'http-trace.json'),
                               'bytes': (output/'http-trace.json').stat().st_size, 'contains_original_excerpts': True}
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['passed'] = bool(checks) and all(item['passed'] for item in checks)
        report['check_count'] = len(checks)
        write(output/'report.json', report)
    return report
