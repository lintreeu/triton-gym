// src/components/KernelTabs.tsx
import * as Tabs from '@radix-ui/react-tabs';
import Editor from '@monaco-editor/react';
import { useState } from 'react';

export type KernelFile = {
  name: string;
  code: string;
};

export interface KernelTabsProps {
  files: KernelFile[];
  onUpdate: (files: KernelFile[]) => void;
}

export default function KernelTabs({ files, onUpdate }: KernelTabsProps) {
  const [activeTab, setActiveTab] = useState(files[0].name);

  const handleChange = (name: string, code: string) => {
    const updated = files.map((f) => (f.name === name ? { ...f, code } : f));
    onUpdate(updated);
  };

  return (
    <Tabs.Root value={activeTab} onValueChange={setActiveTab}>
      <Tabs.List style={{ display: 'flex', borderBottom: '1px solid #ccc' }}>
        {files.map((f) => (
          <Tabs.Trigger
            key={f.name}
            value={f.name}
            style={{
              padding: '8px 12px',
              cursor: 'pointer',
              background: f.name === activeTab ? '#eee' : '#fff',
              border: 'none',
              borderBottom: f.name === activeTab ? '2px solid #333' : 'none'
            }}
          >
            {f.name}
          </Tabs.Trigger>
        ))}
      </Tabs.List>

      {files.map((f) => (
        <Tabs.Content key={f.name} value={f.name} forceMount>
          <Editor
            height="calc(100vh - 100px)"
            defaultLanguage="python"
            theme="vs-light"
            value={f.code}
            onChange={(val) => handleChange(f.name, val || '')}
          />
        </Tabs.Content>
      ))}
    </Tabs.Root>
  );
}