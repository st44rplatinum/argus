import React, { useCallback, useEffect, useState } from 'react';

import { Job, cancelJob, listJobs, submitJob } from '../api';

// Two seconds. Jobs change on the order of minutes, so anything faster is
// noise, and a websocket would be a transport built to watch a short list.
const POLL_MS = 2000;

const STATUS_COLOUR: Record<string, string> = {
  queued: 'text-muted-foreground',
  running: 'text-primary',
  done: 'text-green-500',
  failed: 'text-red-500',
  cancelled: 'text-muted-foreground',
};

function JobRow({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const running = job.status === 'queued' || job.status === 'running';
  const params = job.request as { low?: number; high?: number };

  return (
    <div className="border-border border-b py-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className={`font-medium ${STATUS_COLOUR[job.status] ?? ''}`}>
          {job.status}
          {job.stage && job.status === 'running' ? ` · ${job.stage}` : ''}
        </span>

        {running ? (
          <button
            className="text-muted-foreground hover:text-foreground underline"
            onClick={() => onCancel(job.id)}
          >
            cancel
          </button>
        ) : null}
      </div>

      <div className="text-muted-foreground mt-1 font-mono">
        {params.low} – {params.high} HU
        {job.result ? ` · ${job.result.voxel_count.toLocaleString()} voxels` : ''}
      </div>

      {job.error ? (
        <div className="text-red-500 mt-1 break-words">{job.error.split('\n')[0]}</div>
      ) : null}

      {job.result?.seg_series_uid ? (
        <div className="text-muted-foreground mt-1">
          SEG stored — reload the study to see it
        </div>
      ) : null}
    </div>
  );
}

export default function JobsPanel({ servicesManager }: withAppTypes) {
  const { viewportGridService, displaySetService, uiNotificationService } =
    servicesManager.services;

  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setJobs(await listJobs());
      setError(null);
    } catch (exc) {
      // The service being down is the common case while developing; say so
      // once rather than filling the panel with the same message.
      setError(String(exc));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const activeDisplaySet = useCallback(() => {
    const { activeViewportId, viewports } = viewportGridService.getState();
    const uid = viewports?.get(activeViewportId)?.displaySetInstanceUIDs?.[0];
    return uid ? displaySetService.getDisplaySetByUID(uid) : null;
  }, [viewportGridService, displaySetService]);

  const onSegment = async () => {
    const displaySet = activeDisplaySet();

    if (!displaySet) {
      uiNotificationService?.show({ title: 'Argus', message: 'No active series' });
      return;
    }

    // A SEG is a segmentation, not something to segment.
    if (displaySet.Modality === 'SEG') {
      uiNotificationService?.show({
        title: 'Argus',
        message: 'Select the CT series, not the segmentation',
      });
      return;
    }

    setBusy(true);
    try {
      const job = await submitJob({
        study_uid: displaySet.StudyInstanceUID,
        series_uid: displaySet.SeriesInstanceUID,
        store: true,
      });

      if (job.cached) {
        uiNotificationService?.show({
          title: 'Argus',
          message: 'Already computed — returned from cache',
        });
      }

      await refresh();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  };

  const onCancel = async (id: string) => {
    try {
      await cancelJob(id);
      await refresh();
    } catch (exc) {
      setError(String(exc));
    }
  };

  return (
    <div className="p-3">
      <button
        className="bg-primary text-primary-foreground w-full rounded px-3 py-2 text-xs font-medium disabled:opacity-50"
        onClick={onSegment}
        disabled={busy}
      >
        {busy ? 'Submitting…' : 'Segment bone'}
      </button>

      {error ? (
        <p className="text-red-500 mt-3 break-words text-xs">
          {error.includes('Failed to fetch')
            ? 'AI service unreachable on :8000'
            : error}
        </p>
      ) : null}

      <div className="mt-3">
        {jobs.length === 0 ? (
          <p className="text-muted-foreground text-xs">No jobs yet.</p>
        ) : (
          [...jobs]
            .reverse()
            .map(job => (
              <JobRow
                key={job.id}
                job={job}
                onCancel={onCancel}
              />
            ))
        )}
      </div>
    </div>
  );
}
