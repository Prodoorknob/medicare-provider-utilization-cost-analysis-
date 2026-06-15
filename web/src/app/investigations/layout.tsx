import type { Metadata } from 'next';
import type { ReactNode } from 'react';

// Investigation briefs name real providers beside statistical fraud-suspicion
// signals. They are for authorized analyst review only -- keep the routes out
// of search-engine indexes regardless of deploy config. This is a server
// component so `metadata` applies (the pages themselves are client components).
export const metadata: Metadata = {
  title: 'Investigations',
  robots: { index: false, follow: false, nocache: true },
};

export default function InvestigationsLayout({ children }: { children: ReactNode }) {
  return children;
}
