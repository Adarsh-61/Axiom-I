'use client';

import React from 'react';
import { useRouter } from 'next/navigation';

export default function FeedbackPage() {
  const router = useRouter();

  return (
    <div className="mathPage" style={{ maxWidth: '800px' }}>
      <div className="mathPageHeader" style={{ borderBottom: 'none', paddingBottom: '0' }}>
        <button
          className="backLink"
          onClick={() => {
            if (typeof window !== 'undefined' && window.history.length > 1) {
              router.back();
            } else {
              router.push('/');
            }
          }}
        >
          &#8592; Back to analysis workspace
        </button>
      </div>

      <div className="card" style={{ padding: '0', overflow: 'hidden', border: '1px solid var(--border)', marginTop: '16px' }}>
        <iframe
          src="https://docs.google.com/forms/d/e/1FAIpQLSeQGi-Srw90DQuIjikmieJSs1YR4S7SMxGfksnDQ3AftGlI2Q/viewform?embedded=true"
          width="100%"
          height="1200"
          frameBorder="0"
          marginHeight={0}
          marginWidth={0}
          style={{ border: 'none', display: 'block' }}
        >
          Loading…
        </iframe>
      </div>
    </div>
  );
}
