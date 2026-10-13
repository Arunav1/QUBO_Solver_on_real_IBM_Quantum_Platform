
import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Atom,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Code2,
  FileCheck2,
  FlaskConical,
  LayoutDashboard,
  LoaderCircle,
  Plus,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react'
import './App.css'
import { getJob, optimizeFromText } from './api/client'

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts'


const examples = [
  {
    title: 'Portfolio selection',
    description:
      'Choose a subset of four investment assets to maximize total expected return without exceeding a budget of 100 units. Asset A costs 40 units and returns 60 units. Asset B costs 30 units and returns 45 units. Asset C costs 20 units and returns 32 units. Asset D costs 10 units and returns 14 units. Each asset can either be selected once or not selected. The total cost of selected assets must be at most 100 units.',
    icon: Activity,
  },
  {
    title: 'Task scheduling',
    description:
      'Assign each of three tasks T1, T2, and T3 to exactly one of two machines M1 and M2, minimizing total processing cost. The processing costs are: T1 costs 3 units on M1 and 5 units on M2; T2 costs 4 units on M1 and 2 units on M2; T3 costs 6 units on M1 and 3 units on M2. Each task must be assigned to exactly one machine. Each machine can process multiple tasks.',
    icon: Clock3,
  },
  {
    title: 'Selection problem',
    description:
      'There are two items. Item A costs 3 units and item B costs 2 units. Each item can either be selected or not selected. Select exactly one item and minimize the total cost.',
    icon: Workflow,
  },
]


const JOB_POLL_MS = 3000
const PENDING_JOB_KEY = 'veriqaoa.pendingJob'
const PENDING_STATUSES = ['queued', 'initializing', 'validating', 'running']

function rememberJob(jobId) {
  try {
    if (jobId) localStorage.setItem(PENDING_JOB_KEY, jobId)
    else localStorage.removeItem(PENDING_JOB_KEY)
  } catch {
    // Storage can be unavailable; resuming after reload is optional.
  }
}

function storedJob() {
  try {
    return localStorage.getItem(PENDING_JOB_KEY)
  } catch {
    return null
  }
}

const navigation = [
  { label: 'Dashboard', icon: LayoutDashboard },
  { label: 'Optimize Problem', icon: Sparkles },
  { label: 'Verification', icon: ShieldCheck },
  { label: 'Evaluation', icon: FlaskConical },
]

function App() {
  const [activePage, setActivePage] = useState('Optimize Problem')
  const [problem, setProblem] = useState('')
  const [showExamples, setShowExamples] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [pendingJob, setPendingJob] = useState(null)
  const pollRun = useRef(0)

  // Wait for an IBM Quantum job, refreshing its queue state.
  async function followJob(data) {
    const run = ++pollRun.current
    let current = data

    while (PENDING_STATUSES.includes(current.status)) {
      const job = current.optimization?.ibm_job
      setPendingJob(job)
      rememberJob(job?.job_id)

      await new Promise((resolve) => setTimeout(resolve, JOB_POLL_MS))
      if (run !== pollRun.current) return

      current = await getJob(job.job_id)
      if (run !== pollRun.current) return
    }

    setPendingJob(null)
    rememberJob(null)

    if (current.status === 'failed') {
      throw new Error(current.message || 'The IBM Quantum job failed.')
    }

    setResult(current)
  }

  function stopFollowing() {
    pollRun.current += 1
    setPendingJob(null)
    rememberJob(null)
  }

  // Resume a hardware job that was still queued when the page closed.
  useEffect(() => {
    const jobId = storedJob()
    if (!jobId) return

    setLoading(true)
    getJob(jobId)
      .then(followJob)
      .catch((err) => {
        rememberJob(null)
        setError(err.message)
      })
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])


  async function handleOptimize() {
    if (!problem.trim()) {
      setError('Describe an optimization problem before continuing.')
      return
    }

    stopFollowing()
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await optimizeFromText(problem.trim())
      await followJob(data)
    } catch (err) {
      setError(
        err.message || 'Optimization failed. Check that the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }


  function selectExample(description) {
    setProblem(description)
    setShowExamples(false)
    setError('')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setActivePage('Dashboard')}>
          <span className="brand-mark">
            <Atom size={25} />
          </span>
          <span className="brand-copy">
            <strong>VeriQAOA</strong>
            <small>OPTIMIZATION COMPILER</small>
          </span>
        </button>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${activePage === label ? 'active' : ''}`}
              onClick={() => setActivePage(label)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === 'Verification' && (
                <span className="nav-count">✓</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="backend-card">
            <span className="status-dot" />
            <div>
              <strong>Backend status</strong>
              <small>API connection configured</small>
            </div>
          </div>

          <div className="sidebar-footer">
            <span className="avatar">VQ</span>
            <div className="profile-copy">
              <strong>VeriQAOA</strong>
              <small>Local workspace</small>
            </div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span>
            <span className="crumb-divider">/</span>
            <strong>{activePage}</strong>
          </div>
          <div className="topbar-right">
            <span className="environment-pill">
              <span className="status-dot" />
              Local environment
            </span>
            <button
              className="icon-button"
              aria-label="Help"
              title="Help"
              onClick={() =>
                setError('Describe your problem in plain English to get started.')
              }
            >
              <CircleHelp size={19} />
            </button>
          </div>
        </header>

        <div className="page-container">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">
                <Sparkles size={14} />
                QUANTUM-ASSISTED OPTIMIZATION
              </div>
              <h1>
                Turn complex problems
                <br />
                into <span>verified solutions.</span>
              </h1>
              <p className="welcome-description">
                Formulate, compile, and solve optimization problems with
                AI-assisted modeling and independently verified results.
              </p>
            </div>

            <div className="hero-visual" aria-hidden="true">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="hero-core">
                <Atom size={54} strokeWidth={1.25} />
              </div>
              <span className="orbit-node node-one" />
              <span className="orbit-node node-two" />
              <span className="orbit-node node-three" />
            </div>
          </section>

          <section className="workflow-strip">
            <div className="workflow-step current">
              <span className="step-icon"><Sparkles size={16} /></span>
              <span><strong>01</strong> Formulate</span>
            </div>
            <ArrowRight size={15} className="workflow-arrow" />
            <div className="workflow-step">
              <span className="step-icon"><Code2 size={16} /></span>
              <span><strong>02</strong> Compile to QUBO</span>
            </div>
            <ArrowRight size={15} className="workflow-arrow" />
            <div className="workflow-step">
              <span className="step-icon"><Atom size={16} /></span>
              <span><strong>03</strong> Solve with QAOA</span>
            </div>
            <ArrowRight size={15} className="workflow-arrow" />
            <div className="workflow-step">
              <span className="step-icon"><FileCheck2 size={16} /></span>
              <span><strong>04</strong> Verify</span>
            </div>
          </section>

          <section className="editor-card">
            <div className="card-heading">
              <div className="heading-icon"><Sparkles size={19} /></div>
              <div>
                <h2>Define your optimization problem</h2>
                <p>Describe what you want to optimize. No equations required.</p>
              </div>
              <span className="ai-badge">AI-POWERED</span>
            </div>

            <label htmlFor="problem-input" className="input-label">
              PROBLEM DESCRIPTION
            </label>
            <textarea
              id="problem-input"
              value={problem}
              onChange={(event) => {
                setProblem(event.target.value)
                setError('')
              }}
              placeholder="Example: I have five projects with different costs and expected returns. Select the best combination without exceeding my budget..."
              rows={5}
            />

            <div className="editor-bottom">
              <span className="character-count">
                {problem.length} characters
              </span>
              <button
                className="text-button"
                onClick={() => {
                  stopFollowing()
                  setLoading(false)
                  setProblem('')
                  setError('')
                  setResult(null)
                }}
              >
                Clear input
              </button>
            </div>

            {pendingJob && (
              <div className="notice job-notice">
                <LoaderCircle size={17} className="spin" />
                <span>
                  Running on IBM Quantum <strong>{pendingJob.backend}</strong>{' '}
                  · job <code>{pendingJob.job_id}</code> is{' '}
                  {pendingJob.state?.toLowerCase() ?? 'queued'}.
                  {pendingJob.pending_jobs_at_submit != null &&
                    ` ${pendingJob.pending_jobs_at_submit} jobs were ahead in the queue at submission.`}{' '}
                  Results appear here automatically; you can leave this page open or come back later.
                </span>
              </div>
            )}

            {error && (
              <div className="notice">
                <CircleHelp size={17} />
                <span>{error}</span>
              </div>
            )}

            <div className="editor-actions">
              <div className="action-hint">
                <ShieldCheck size={16} />
                Independent solution verification
              </div>
              <button
                className="primary-button"
                onClick={handleOptimize}
                disabled={loading}
              >
                {loading ? (
                  <LoaderCircle size={17} className="spin" />
                ) : (
                  <Sparkles size={17} />
                )}
                Optimize problem
                <ArrowRight size={17} />
              </button>
            </div>
          </section>

          {showExamples && (
            <section className="examples-section">
              <div className="section-heading">
                <div>
                  <h2>Need a starting point?</h2>
                  <p>Try an example problem to explore the workflow.</p>
                </div>
                <button
                  className="text-button"
                  onClick={() => setShowExamples(false)}
                >
                  Hide examples
                </button>
              </div>

              <div className="example-grid">
                {examples.map(({ title, description, icon: Icon }) => (
                  <button
                    className="example-card"
                    key={title}
                    onClick={() => selectExample(description)}
                  >
                    <span className="example-icon"><Icon size={19} /></span>
                    <span className="example-content">
                      <strong>{title}</strong>
                      <span>{description}</span>
                    </span>
                    <Plus size={18} className="example-plus" />
                  </button>
                ))}
              </div>
            </section>
          )}


          {result && (() => {
            const optimization = result.optimization
            const verification = optimization?.solution_verification
            const qaoa = optimization?.qaoa

            const bestAssignment =
              qaoa?.best_sample?.decision_assignment ?? qaoa?.best_sample?.assignment ?? {}

            // Keep the chart readable for larger problems: show the
            // most frequently measured assignments only.
            const chartData = (qaoa?.samples ?? [])
              .slice()
              .sort((a, b) => b.count - a.count)
              .slice(0, 24)
              .map((sample) => ({
                assignment: Object.entries(sample.assignment)
                  .map(([name, value]) => `${name}=${value}`)
                  .join(', '),
                count: sample.count,
                energy: sample.energy,
                probability: sample.probability,
                isBest: Object.entries(bestAssignment).every(
                  ([name, value]) => sample.assignment[name] === value
                ),
              }))

            const assignment = verification?.assignment ?? qaoa?.best_sample?.assignment ?? {}
            const objective = verification?.original_objective
            const quboVerification = optimization?.final_verification
            const variables = Object.entries(assignment)



            return (
              <section className="results-dashboard">
                <div className="results-heading">
                  <div>
                    <div className="eyebrow">
                      <CheckCircle2 size={15} />
                      OPTIMIZATION REPORT
                    </div>
                    <h2>{result.formulation?.title ?? 'Optimization result'}</h2>
                    <p>Your solution and verification results</p>
                  </div>
                  <span className={`result-status ${verification?.verified ? 'success' : 'warning'}`}>
                    {verification?.verified ? 'Verified' : 'Review required'}
                  </span>
                </div>

                <div className="metric-grid">
                  <article className="metric-card">
                    <span>Objective value</span>
                    <strong>{objective ?? '—'}</strong>
                    <small>{result.formulation?.model?.sense === 'max' ? 'Maximization' : 'Minimization'}</small>
                  </article>

                  <article className="metric-card">
                    <span>Variables</span>
                    <strong>{variables.length}</strong>
                    <small>Binary assignments</small>
                  </article>

                  <article className="metric-card">
                    <span>QUBO equivalence</span>
                    <strong>{quboVerification?.equivalent ? 'Passed' : 'Not confirmed'}</strong>
                    <small>{quboVerification?.assignments_checked ?? 0} assignments checked</small>
                  </article>

                  <article className="metric-card">
                    <span>Solution verification</span>
                    <strong>{verification?.verified && verification?.feasible ? 'Passed' : 'Review'}</strong>
                    <small>{verification?.optimal ? 'Optimality confirmed' : 'Optimality not confirmed'}</small>
                  </article>
                </div>

                <div className="result-section">
                  <h3>Variable assignments</h3>
                  <div className="assignment-list">
                    {variables.map(([name, value]) => (
                      <div className="assignment-row" key={name}>
                        <code>{name}</code>
                        <span>{value === 1 ? 'Selected' : 'Not selected'}</span>
                        <strong className={value === 1 ? 'value-selected' : ''}>{value}</strong>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="result-section sampling-section">
                  <div className="sampling-heading">
                    <div>
                      <h3>QAOA sampling distribution</h3>
                      <p>
                        Frequency of each measured assignment across{' '}
                        {qaoa?.shots?.toLocaleString() ?? 0} shots
                      </p>
                    </div>

                    <span className="best-sample-badge">
                      Best energy: {qaoa?.best_sample?.energy ?? '—'}
                    </span>
                  </div>

                  {chartData.length > 0 ? (
                    <div className="sampling-chart">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={chartData}
                          margin={{ top: 20, right: 12, left: 0, bottom: 12 }}
                        >
                          <CartesianGrid
                            stroke="#293143"
                            strokeDasharray="3 3"
                            vertical={false}
                          />

                          <XAxis
                            dataKey="assignment"
                            tick={{ fill: '#aab4c8', fontSize: 11 }}
                            axisLine={{ stroke: '#343d50' }}
                            tickLine={false}
                            interval={0}
                          />

                          <YAxis
                            allowDecimals={false}
                            tick={{ fill: '#aab4c8', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            label={{
                              value: 'Shots',
                              angle: -90,
                              position: 'insideLeft',
                              fill: '#aab4c8',
                              fontSize: 12,
                            }}
                          />

                          <Tooltip
                            cursor={{ fill: 'rgba(140, 229, 245, 0.06)' }}
                            contentStyle={{
                              background: '#151b27',
                              border: '1px solid #343d50',
                              borderRadius: 10,
                              color: '#e8edf7',
                            }}
                            labelStyle={{ color: '#8ce5f5', marginBottom: 8 }}
                            formatter={(value, name, props) => {
                              const row = props.payload

                              if (name === 'count') {
                                return [
                                  `${value} shots (${(row.probability * 100).toFixed(2)}%)`,
                                  'Frequency',
                                ]
                              }

                              return [value, name]
                            }}
                          />

                          <Bar
                            dataKey="count"
                            name="Frequency"
                            radius={[6, 6, 0, 0]}
                            maxBarSize={72}
                          >
                            {chartData.map((entry) => (
                              <Cell
                                key={entry.assignment}
                                fill={entry.isBest ? '#52dfb1' : '#62cce8'}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="chart-empty">
                      No sampling data is available for this optimization run.
                    </p>
                  )}

                  <div className="sampling-legend">
                    <span>
                      <i className="legend-dot best-dot" />
                      Lowest-energy sample
                    </span>
                    <span>
                      <i className="legend-dot normal-dot" />
                      Other samples
                    </span>
                  </div>

                  <div className="sample-table-wrapper">
                    <table className="sample-table">
                      <thead>
                        <tr>
                          <th>Assignment</th>
                          <th>Shots</th>
                          <th>Probability</th>
                          <th>Energy</th>
                          <th>Result</th>
                        </tr>
                      </thead>

                      <tbody>
                        {chartData
                          .slice()
                          .sort((a, b) => a.energy - b.energy)
                          .map((sample) => (
                            <tr key={sample.assignment}>
                              <td><code>{sample.assignment}</code></td>
                              <td>{sample.count.toLocaleString()}</td>
                              <td>{(sample.probability * 100).toFixed(2)}%</td>
                              <td>{sample.energy}</td>
                              <td>
                                {sample.isBest ? (
                                  <span className="sample-best-label">Best sample</span>
                                ) : (
                                  <span className="sample-other-label">Sampled</span>
                                )}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>


                <div className="result-section">
                  <h3>QAOA execution</h3>
                  <div className="detail-list">
                    <div><span>Solver</span><strong>{qaoa?.solver ?? '—'}</strong></div>
                    <div><span>Backend</span><strong>{qaoa?.backend ?? '—'}</strong></div>
                    <div><span>QAOA layers</span><strong>{qaoa?.layers ?? '—'}</strong></div>
                    <div><span>Shots</span><strong>{qaoa?.shots?.toLocaleString() ?? '—'}</strong></div>
                    <div><span>Optimizer iterations</span><strong>{qaoa?.optimizer?.iterations ?? '—'}</strong></div>
                    <div><span>Best sample energy</span><strong>{qaoa?.best_sample?.energy ?? '—'}</strong></div>
                    {qaoa?.ibm_job && (
                      <>
                        <div><span>IBM job ID</span><strong>{qaoa.ibm_job.job_id}</strong></div>
                        <div><span>Transpiled depth</span><strong>{qaoa.ibm_job.transpiled_depth ?? '—'}</strong></div>
                        <div><span>Two-qubit gates</span><strong>{qaoa.ibm_job.two_qubit_gates ?? '—'}</strong></div>
                        <div><span>QPU time</span><strong>{qaoa.ibm_job.quantum_seconds != null ? `${qaoa.ibm_job.quantum_seconds} s` : '—'}</strong></div>
                      </>
                    )}
                  </div>
                </div>

                <details className="technical-details">
                  <summary>View QUBO model and raw API response</summary>
                  <h3>QUBO terms</h3>
                  <pre>{JSON.stringify(optimization?.qubo, null, 2)}</pre>
                  <h3>Complete response</h3>
                  <pre>{JSON.stringify(result, null, 2)}</pre>
                </details>
              </section>
            )
          })()}


          <footer className="page-footer">
            <span>VeriQAOA <span className="footer-separator">·</span> Research prototype</span>
            <span>AI formulation <span className="footer-separator">·</span> QUBO <span className="footer-separator">·</span> QAOA <span className="footer-separator">·</span> Verification</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
