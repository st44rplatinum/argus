import { useCallback, useEffect, useRef, useState } from 'react';

import { Job, getJob, submitJob } from './api';

const POLL_MS = 2000;

type Params = Parameters<typeof submitJob>[0];

/** Submit one job and follow it until it finishes. */
export default function useJob() {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = () => {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
  };

  useEffect(() => stop, []);

  const submit = useCallback(async (params: Params) => {
    stop();
    setError(null);
    setJob(null);

    try {
      const submitted = await submitJob(params);
      setJob(submitted);
      if (submitted.status !== 'queued' && submitted.status !== 'running') {
        return;
      }

      timer.current = setInterval(async () => {
        try {
          const next = await getJob(submitted.id);
          setJob(next);
          if (next.status !== 'queued' && next.status !== 'running') {
            stop();
          }
        } catch (exc) {
          setError(String(exc));
          stop();
        }
      }, POLL_MS);
    } catch (exc) {
      setError(String(exc));
    }
  }, []);

  const running = job?.status === 'queued' || job?.status === 'running';
  // a traceback's last line is the actual error
  const failure = job?.status === 'failed' ? job.error?.split('\n').filter(Boolean).pop() : null;

  return { job, submit, running, error: error ?? failure ?? null, setError };
}
