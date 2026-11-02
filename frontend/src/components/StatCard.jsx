import React from 'react';

export const StatCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  trend,
  color = 'cyan', // cyan, emerald, amber, rose, purple
}) => {
  const colorMap = {
    cyan: {
      border: 'border-cyan-500/20 hover:border-cyan-500/40',
      iconBg: 'bg-cyan-500/10 text-cyan-400',
      accent: 'text-cyan-400',
    },
    emerald: {
      border: 'border-emerald-500/20 hover:border-emerald-500/40',
      iconBg: 'bg-emerald-500/10 text-emerald-400',
      accent: 'text-emerald-400',
    },
    amber: {
      border: 'border-amber-500/20 hover:border-amber-500/40',
      iconBg: 'bg-amber-500/10 text-amber-400',
      accent: 'text-amber-400',
    },
    rose: {
      border: 'border-rose-500/20 hover:border-rose-500/40',
      iconBg: 'bg-rose-500/10 text-rose-400',
      accent: 'text-rose-400',
    },
    purple: {
      border: 'border-purple-500/20 hover:border-purple-500/40',
      iconBg: 'bg-purple-500/10 text-purple-400',
      accent: 'text-purple-400',
    },
  };

  const scheme = colorMap[color] || colorMap.cyan;

  return (
    <div
      className={`glass-panel rounded-xl p-4 sm:p-5 border transition-all duration-200 ${scheme.border} relative overflow-hidden`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-[11px] sm:text-xs font-medium text-slate-400 uppercase tracking-wider mb-1 truncate">
            {title}
          </p>
          <div className="flex items-baseline flex-wrap gap-1.5 sm:gap-2">
            <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight break-words">{value}</h3>
            {badge && <div className="shrink-0">{badge}</div>}
          </div>
          {subtitle && (
            <p className="text-[11px] sm:text-xs text-slate-400 mt-1 flex flex-wrap items-center gap-1 leading-snug break-words">
              {trend && (
                <span
                  className={
                    trend.isPositive ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'
                  }
                >
                  {trend.value}
                </span>
              )}
              <span>{subtitle}</span>
            </p>
          )}
        </div>
        {Icon && (
          <div className={`p-2.5 sm:p-3 rounded-lg shrink-0 ${scheme.iconBg}`}>
            <Icon className="w-4 h-4 sm:w-5 sm:h-5" />
          </div>
        )}
      </div>
    </div>
  );
};

export default StatCard;
