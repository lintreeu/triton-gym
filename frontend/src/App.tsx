// src/App.tsx
import { useState } from 'react';
import KernelTabs, { type KernelFile } from './components/KernelTabs';
import { TritonMemoryView, type ViewMode } from './components/TritonMemoryView';
import { MemoryLegend } from './components/MemoryLegend';

const defaultCode = `import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)`;

function App() {
  const [files, setFiles] = useState<KernelFile[]>([
    { name: 'main.triton', code: defaultCode },
    { name: 'matmul.triton', code: '// define matmul here...' }
  ]);

  const [output, setOutput] = useState<string>('');
  const [vectorX, setVectorX] = useState<number[]>([]);
  const [vectorY, setVectorY] = useState<number[]>([]);
  const [vectorZ, setVectorZ] = useState<number[]>([]);
  const [error, setError] = useState<string>('');
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const [step, setStep] = useState<number>(Infinity);
  const [log, setLog] = useState<string>('');
  const N = 512;
  const BLOCK_SIZE = 128;

  const handleRun = async () => {
    const currentCode = files[0].code;
    setLog((prev) => prev + '▶️ Running current kernel...\n');
    try {
      const res = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: currentCode, size: N })
      });
      const data = await res.json();
      if (res.ok) {
        setOutput(JSON.stringify(data.result));
        setVectorX(new Array(N).fill(1));
        setVectorY(new Array(N).fill(2));
        setVectorZ(data.result);
        setStep(1);
        setError('');
        setLog((prev) => prev + '✅ Kernel execution completed.\n');
      } else {
        throw new Error(data.detail || '執行錯誤');
      }
    } catch (err: any) {
      setError(err.message);
      setLog((prev) => prev + `❌ Error: ${err.message}\n`);
    }
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        height: '100vh',
        overflow: 'hidden',
        fontFamily: 'sans-serif'
      }}
    >
      {/* 左側：多 Tab Kernel 編輯 */}
      <div
        style={{
          padding: '1rem',
          borderRight: '1px solid #ccc',
          display: 'flex',
          flexDirection: 'column',
          height: '100%'
        }}
      >
        <KernelTabs files={files} onUpdate={setFiles} />
      </div>

      {/* 右側：輸出 + 控制列 + 記憶體圖表 */}
      <div
        style={{
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflowY: 'auto'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3>Output</h3>
          <button
            onClick={handleRun}
            style={{ backgroundColor: '#d33', color: '#fff', padding: '4px 12px', borderRadius: '4px' }}
          >
            RUN ▶
          </button>
        </div>

        <div
          style={{
            background: '#f5f5f5',
            padding: '12px',
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            maxHeight: '120px',
            overflowY: 'auto',
            borderRadius: '4px',
            flexShrink: 0
          }}
        >
          {error ? <span style={{ color: 'red' }}>{error}</span> : `Output: ${output}`}
        </div>

        <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <label><strong>顯示模式:</strong></label>
          <select value={viewMode} onChange={e => setViewMode(e.target.value as ViewMode)}>
            <option value="all">全部</option>
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
          <button onClick={() => setStep((prev) => Math.max(1, prev - 1))}>⬅️ 上一步</button>
          <button onClick={() => setStep((prev) => prev === Infinity ? 1 : prev + 1)}>➡️ 下一步</button>
        </div>

        <div style={{ marginTop: '1rem' }}>
          <TritonMemoryView
            X={vectorX}
            Y={vectorY}
            Z={vectorZ}
            N={N}
            BLOCK_SIZE={BLOCK_SIZE}
            mode={viewMode}
            step={step}
          />
          <MemoryLegend />
        </div>

        <div style={{ marginTop: '1rem' }}>
          <h3>Console</h3>
          <div
            style={{
              background: '#000',
              color: '#0f0',
              fontFamily: 'monospace',
              padding: '0.5rem',
              borderRadius: '4px',
              height: '100px',
              overflowY: 'auto',
              fontSize: '13px'
            }}
          >
            <pre>{log}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;