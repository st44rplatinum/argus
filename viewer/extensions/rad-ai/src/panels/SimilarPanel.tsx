import React, { useCallback, useEffect, useRef, useState } from 'react';

import { Job, SimilarMatch, getJob, submitJob } from '../api';

const POLL_MS = 2000;

const ORGANS: [string, string][] = [
  ['liver', 'Liver'],
  ['spleen', 'Spleen'],
  ['kidney_left', 'Left kidney'],
  ['kidney_right', 'Right kidney'],
  ['aorta', 'Aorta'],
  ['heart', 'Heart'],
];

const FEATURES: Record<string, string> = {
  mean_hu: 'density',
  std_hu: 'density spread',
  p10_hu: 'low density',
  p90_hu: 'high density',
  skew_hu: 'density skew',
  volume_ml: 'volume',
  surface_to_volume: 'surface',
  elongation: 'elongation',
  flatness: 'flatness',
  gradient_mean: 'texture',
  heterogeneity: 'heterogeneity',
};

const phaseLabel = (phase: string | null) => (phase ? phase.replace('_', ' ') : 'unknown phase');

function MatchRow({ match }: { match: SimilarMatch }) {
  // Own tab, so the query stays open to compare against.
  const open = () =>
    window.open(`${window.location.origin}/rad-ai?StudyInstanceUIDs=${match.study_uid}`, '_blank');

  return (
    <div className="border-border border-b py-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium">{match.case}</span>
        <span className="font-mono">{match.similarity.toFixed(3)}</span>
      </div>

      <div className="text-muted-foreground mt-1">
        {phaseLabel(match.phase)}
        {match.reasons.length
          ? ` · partly because: ${match.reasons.map(r => FEATURES[r] ?? r).join(', ')}`
          : ''}
      </div>

      <button
        className="text-primary mt-1 underline"
        onClick={open}
      >
        open study
      </button>
    </div>
  );
}

export default function SimilarPanel({ servicesManager }: withAppTypes) {
  const { viewportGridService, displaySetService } = servicesManager.services;

  const [organ, setOrgan] = useState('liver');
  const [phaseCentred, setPhaseCentred] = useState(false);
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

  const poll = useCallback((id: string) => {
    stop();
    timer.current = setInterval(async () => {
      try {
        const next = await getJob(id);
        setJob(next);
        if (next.status !== 'queued' && next.status !== 'running') {
          stop();
        }
      } catch (exc) {
        setError(String(exc));
        stop();
      }
    }, POLL_MS);
  }, []);

  const onFind = async () => {
    const { activeViewportId, viewports } = viewportGridService.getState();
    const uid = viewports?.get(activeViewportId)?.displaySetInstanceUIDs?.[0];
    const displaySet = uid ? displaySetService.getDisplaySetByUID(uid) : null;

    if (!displaySet || displaySet.Modality !== 'CT') {
      setError('Select a CT series first');
      return;
    }

    setError(null);
    setJob(null);

    try {
      const submitted = await submitJob({
        study_uid: displaySet.StudyInstanceUID,
        series_uid: displaySet.SeriesInstanceUID,
        model: 'similar',
        organ,
        k: 5,
        phase_centred: phaseCentred,
      });
      setJob(submitted);
      if (submitted.status === 'queued' || submitted.status === 'running') {
        poll(submitted.id);
      }
    } catch (exc) {
      setError(String(exc));
    }
  };

  const running = job?.status === 'queued' || job?.status === 'running';
  const matches: SimilarMatch[] = job?.status === 'done' ? job.result?.results ?? [] : [];

  return (
    <div className="text-foreground p-3 text-xs">
      <label className="text-muted-foreground block">Organ</label>
      <select
        className="bg-background text-foreground border-border mt-1 w-full rounded border px-2 py-1"
        value={organ}
        onChange={e => setOrgan(e.target.value)}
        disabled={running}
      >
        {ORGANS.map(([value, label]) => (
          <option
            key={value}
            value={value}
          >
            {label}
          </option>
        ))}
      </select>

      <label className="mt-2 flex items-center gap-2">
        <input
          type="checkbox"
          checked={phaseCentred}
          onChange={e => setPhaseCentred(e.target.checked)}
          disabled={running}
        />
        Ignore contrast phase
      </label>

      <button
        className="bg-primary text-primary-foreground mt-3 w-full rounded px-3 py-2 font-medium disabled:opacity-50"
        onClick={onFind}
        disabled={running}
      >
        {running ? `Searching… ${job?.stage ?? job?.status}` : 'Find similar'}
      </button>

      {error ? (
        <p className="mt-3 break-words text-red-500">
          {error.includes('Failed to fetch') ? 'AI service unreachable on :8000' : error}
        </p>
      ) : null}

      {job?.status === 'failed' ? (
        <p className="mt-3 break-words text-red-500">{job.error?.split('\n').filter(Boolean).pop()}</p>
      ) : null}

      {job?.status === 'done' ? (
        <div className="mt-3">
          <p className="text-muted-foreground">
            This scan: {phaseLabel(job.result?.phase)}
            {job.cached ? ' · cached' : ''}
          </p>
          {matches.length === 0 ? (
            <p className="text-muted-foreground mt-2">No matches in the pool.</p>
          ) : (
            matches.map(m => (
              <MatchRow
                key={m.case}
                match={m}
              />
            ))
          )}
          <p className="text-muted-foreground mt-2">
            Reasons cover the measured features only; the image model's part has no names.
          </p>
        </div>
      ) : null}
    </div>
  );
}
