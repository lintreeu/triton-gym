// src/components/TritonMemoryView.tsx
import * as d3 from 'd3';
import { useEffect } from 'react';

export type ViewMode = 'all' | 'x' | 'y' | 'z';

export interface TritonMemoryProps {
    X: number[];
    Y: number[];
    Z: number[];
    N: number;
    BLOCK_SIZE: number;
    mode: ViewMode;
    step?: number; // 動畫用：只顯示到第幾個 block
}

export function TritonMemoryView({ X, Y, Z, N, BLOCK_SIZE, mode, step = Infinity }: TritonMemoryProps) {
    useEffect(() => {
        const svg = d3.select('#viz')
            .attr('width', 800)
            .attr('height', 300)
            .style('background-color', '#fff');

        svg.selectAll('*').remove();

        const tooltip = d3.select('#tooltip');
        const barSize = 10;
        const padding = 1;
        const blocks = Math.ceil(N / BLOCK_SIZE);
        const data = [];

        for (let pid = 0; pid < blocks; pid++) {
            if (pid >= step) continue;
            for (let i = 0; i < BLOCK_SIZE; i++) {
                const offset = pid * BLOCK_SIZE + i;
                const valid = offset < N;
                data.push({ pid, offset, valid });
            }
        }

        const colorMap = (offset: number, valid: boolean): string => {
            if (!valid) return 'black';
            if (mode === 'x') return offset < X.length ? '#555' : '#eee';
            if (mode === 'y') return offset < Y.length ? '#aaa' : '#eee';
            if (mode === 'z') return offset < Z.length ? '#ffd700' : '#eee';
            if (offset < X.length) return '#555';
            if (offset < Y.length) return '#aaa';
            if (offset < Z.length) return '#ffd700';
            return '#ccc';
        };

        svg.selectAll('rect')
            .data(data)
            .enter()
            .append('rect')
            .attr('x', (d) => (d.offset % 64) * (barSize + padding))
            .attr('y', (d) => d.pid * (barSize + padding))
            .attr('width', barSize)
            .attr('height', barSize)
            .attr('fill', (d) => colorMap(d.offset, d.valid))
            .on('mouseover', (event, d) => {
                tooltip.style('display', 'block')
                    .style('left', event.pageX + 10 + 'px')
                    .style('top', event.pageY + 10 + 'px')
                    .text(`offset: ${d.offset}, block: ${d.pid}`);
            })
            .on('mouseout', () => tooltip.style('display', 'none'));
    }, [X, Y, Z, N, BLOCK_SIZE, mode, step]);

    return (
        <>
            <svg id="viz"></svg>
            <div id="tooltip" style={{
                position: 'fixed', // ❗ 避免滑動跑位
                zIndex: 10,
                display: 'none',
                padding: '4px 8px',
                backgroundColor: '#333',
                color: '#fff',
                borderRadius: '4px',
                fontSize: '12px',
                pointerEvents: 'none'
            }}></div>
        </>
    );
}