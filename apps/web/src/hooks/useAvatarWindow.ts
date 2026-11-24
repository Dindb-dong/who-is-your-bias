import { useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';

type PictureInPictureWindow = Window & {
  documentPictureInPicture?: { requestWindow(options: { width: number; height: number }): Promise<Window> };
};

function prepareWindow(view: Window, wideCaptions: boolean): HTMLElement {
  const doc = view.document;
  doc.title = 'Presentation avatar';
  doc.documentElement.lang = document.documentElement.lang || 'en';
  doc.body.classList.add('avatar-window');
  doc.body.classList.toggle('avatar-window-wide', wideCaptions);

  for (const sheet of document.styleSheets) {
    try {
      const style = doc.createElement('style');
      style.media = sheet.media.mediaText;
      style.textContent = Array.from(sheet.cssRules, (rule) => rule.cssText).join('\n');
      doc.head.append(style);
    } catch {
      if (!sheet.href) continue;
      const link = doc.createElement('link');
      link.rel = 'stylesheet';
      link.href = sheet.href;
      link.media = sheet.media.mediaText;
      doc.head.append(link);
    }
  }

  const target = doc.createElement('main');
  target.className = 'avatar-window-stage';
  const disclosure = doc.createElement('p');
  disclosure.className = 'avatar-window-disclosure';
  disclosure.textContent = 'AI simulation';
  doc.body.append(target, disclosure);
  return target;
}

export function useAvatarWindow(wideCaptions: boolean) {
  const [target, setTarget] = useState<HTMLElement | null>(null);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState('');
  const attempt = useRef<AbortController | null>(null);
  const active = useRef<{ view: Window; removeCloseListener: () => void } | null>(null);
  const pip = (window as PictureInPictureWindow).documentPictureInPicture;

  function close() {
    attempt.current?.abort();
    attempt.current = null;
    setOpening(false);
    active.current?.view.close();
  }

  async function open() {
    if (attempt.current || active.current) return;
    const controller = new AbortController();
    attempt.current = controller;
    setOpening(true);
    setError('');
    let view: Window | null = null;
    try {
      const { width, height } = wideCaptions ? { width: 760, height: 380 } : { width: 420, height: 320 };
      view = pip ? await pip.requestWindow({ width, height })
        : window.open('', '_blank', `popup,width=${width},height=${height}`);
      if (controller.signal.aborted) { view?.close(); return; }
      if (!view || view.closed) throw new Error('Window unavailable');

      const presentation = prepareWindow(view, wideCaptions);
      const onClose = () => {
        active.current = null;
        // Move the existing canvas back before the browser tears down this document.
        flushSync(() => setTarget(null));
      };
      view.addEventListener('pagehide', onClose, { once: true });
      active.current = { view, removeCloseListener: () => view?.removeEventListener('pagehide', onClose) };
      setTarget(presentation);
    } catch {
      view?.close();
      if (!controller.signal.aborted) {
        setError(pip ? 'Could not open picture-in-picture. Try opening it again.'
          : 'Could not open the avatar window. Allow pop-ups for this site and try again.');
      }
    } finally {
      if (attempt.current === controller) {
        attempt.current = null;
        setOpening(false);
      }
    }
  }

  useEffect(() => {
    const dispose = () => {
      attempt.current?.abort();
      attempt.current = null;
      active.current?.removeCloseListener();
      active.current?.view.close();
      active.current = null;
    };
    const onPageHide = () => {
      // Restore the inline stage if the opener is later resumed from the back/forward cache.
      if (active.current || attempt.current) flushSync(() => {
        setTarget(null);
        setOpening(false);
      });
      dispose();
    };
    window.addEventListener('pagehide', onPageHide);
    return () => {
      window.removeEventListener('pagehide', onPageHide);
      dispose();
    };
  }, []);

  return { target, opening, error, pip: Boolean(pip), open, close };
}
