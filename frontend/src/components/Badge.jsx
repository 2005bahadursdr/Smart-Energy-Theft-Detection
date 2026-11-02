import React from 'react';

export const Badge = ({ children, variant = 'default', className = '' }) => {
  const variantStyles = {
    default: 'bg-slate-800 text-slate-300 border-slate-700',
    primary: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
    success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    warning: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    danger: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    blockchain: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
    tamperOk: 'bg-emerald-500/20 text-emerald-300 border-emerald-400/50 font-mono font-bold tracking-wider',
    tamperFail: 'bg-rose-500/20 text-rose-300 border-rose-400/50 font-mono font-bold tracking-wider',
    demo: 'bg-amber-500/15 text-amber-300 border-amber-500/40 text-xs font-mono font-semibold tracking-wide',
    groundTruth: 'bg-rose-500/15 text-rose-300 border-rose-500/40 text-xs font-mono font-semibold tracking-wide',
  };

  const selected = variantStyles[variant] || variantStyles.default;

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${selected} ${className}`}
    >
      {children}
    </span>
  );
};

export default Badge;
