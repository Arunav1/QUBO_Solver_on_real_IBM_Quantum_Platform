async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message =
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data);

    throw new Error(message || `Request failed (${response.status})`);
  }

  return data;
}

export async function checkBackend() {
  return request("/health");
}

export async function optimizeFromText(description) {
  return request("/api/optimize-from-text", {
    method: "POST",
    body: JSON.stringify({
      description,
      p: 1,
      shots: 2048,
      maxiter: 100,
      max_variables: 20,
      tolerance: 1e-8,
    }),
  });
}

export async function getJob(jobId) {
  return request(`/api/jobs/${encodeURIComponent(jobId)}`);
}
