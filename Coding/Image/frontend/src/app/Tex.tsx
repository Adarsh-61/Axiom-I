'use client';

import katex from 'katex';
import 'katex/dist/katex.min.css';

export function Tex({ math, block = false }: { math: string; block?: boolean }) {
  const html = katex.renderToString(math, {
    throwOnError: false,
    displayMode: block,
    trust: true,
  });
  const Tag = block ? 'div' : 'span';
  return <Tag dangerouslySetInnerHTML={{ __html: html }} style={block ? { overflowX: 'auto', padding: '4px 0' } : undefined} />;
}
