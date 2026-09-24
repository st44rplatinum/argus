// Thin client for the AI service. The viewer never imports torch and the
// service never renders; this is the whole contract between them.

const BASE = 'http://localhost:8000';

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export type Job = {
  id: string;
  status: JobStatus;
  stage: string | null;
  error: string | null;
  result: {
    voxel_count: number;
    seg_series_uid: string | null;
  } | null;
  request: Record<string, unknown>;
  created_at: string;
  cached?: boolean;
};

async function json(response: Response) {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status}: ${body.slice(0, 200)}`);
  }
  return response.json();
}

export async function health() {
  return json(await fetch(`${BASE}/health`));
}

export async function submitJob(params: {
  study_uid: string;
  series_uid: string;
  low?: number;
  high?: number;
  store?: boolean;
}): Promise<Job> {
  return json(
    await fetch(`${BASE}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
  );
}

export async function listJobs(): Promise<Job[]> {
  return json(await fetch(`${BASE}/jobs`));
}

export async function cancelJob(jobId: string): Promise<Job> {
  return json(await fetch(`${BASE}/jobs/${jobId}/cancel`, { method: 'POST' }));
}
