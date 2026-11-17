import { useEffect, useState } from 'react';
import { parseGestureBasis, type GestureBasisModel } from '../avatar/learnedGesture';

export function useLearnedGesture(enabled: boolean, personaId?: string, bundle = 'jensen-gesture-basis') {
  const [model, setModel] = useState<{ bundle: string; value: GestureBasisModel }>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    setModel(undefined); setError(undefined);
    if (!enabled || personaId !== 'jensen-huang') { setLoading(false); return; }
    if (!/^[a-zA-Z0-9_-]{1,80}$/.test(bundle)) {
      setLoading(false); setError('Choose a valid local gesture bundle name.'); return;
    }
    const controller = new AbortController(); setLoading(true);
    void (async () => {
      try {
        const response = await fetch(`/models/gesture/${bundle}.json`, { signal: controller.signal });
        if (!response.ok) throw new Error('Install the local Jensen gesture model to enable learned movement.');
        const text = await response.text();
        if (text.length > 4 * 1024 * 1024) throw new Error('The local gesture model exceeds the supported size.');
        const parsed = parseGestureBasis(JSON.parse(text));
        if (!controller.signal.aborted) setModel({ bundle, value: parsed });
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'The local gesture model could not be loaded.');
      } finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  }, [enabled, personaId, bundle]);
  // Effect cleanup follows render; subject isolation must apply immediately.
  return { model: enabled && personaId === 'jensen-huang' && model?.bundle === bundle ? model.value : undefined, error, loading };
}
