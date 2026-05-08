/* ══════════════════════════════════════════
   AI Loan Default Predictor — script.js
   ══════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  // ── FORM LOADING STATE ──
  const form      = document.getElementById('loanForm');
  const submitBtn = document.getElementById('submitBtn');
  const btnText   = document.getElementById('btnText');

  if (form && submitBtn) {
    form.addEventListener('submit', (e) => {
      // Basic validation
      const income = parseFloat(document.getElementById('income')?.value);
      const loan   = parseFloat(document.getElementById('loan_amount')?.value);
      const fico   = parseFloat(document.getElementById('credit_score')?.value);

      if (income <= 0 || isNaN(income)) {
        e.preventDefault();
        showFieldError('income', 'Enter a valid income');
        return;
      }
      if (loan <= 0 || isNaN(loan)) {
        e.preventDefault();
        showFieldError('loan_amount', 'Enter a valid loan amount');
        return;
      }
      if (fico < 300 || fico > 850 || isNaN(fico)) {
        e.preventDefault();
        showFieldError('credit_score', 'Credit score must be between 300 and 850');
        return;
      }

      // Show loading
      submitBtn.disabled = true;
      btnText.textContent = 'Analyzing…';
      submitBtn.style.opacity = '0.75';
    });
  }

  // ── FIELD ERROR HIGHLIGHT ──
  function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    if (!field) return;
    field.style.borderColor = 'var(--danger)';
    field.style.background  = '#fff5f5';
    field.focus();

    let hint = field.parentElement.querySelector('.form-hint');
    if (hint) {
      hint.style.color = 'var(--danger)';
      hint.textContent = '⚠ ' + message;
    }

    field.addEventListener('input', () => {
      field.style.borderColor = '';
      field.style.background  = '';
      if (hint) {
        hint.style.color = '';
        hint.textContent = '';
      }
    }, { once: true });
  }

  // ── PROBABILITY BAR ANIMATION ──
  // Trigger CSS transition after page load on result page
  const probBar = document.querySelector('.prob-bar-fill');
  if (probBar) {
    const target = probBar.dataset.target || probBar.style.width;
    probBar.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        probBar.style.width = parseFloat(target) + '%';
      });
    });
  }

  // ── ANIMATE SHAP BARS ──
  document.querySelectorAll('.shap-bar-fill, .model-bar-fill, .drift-bar-fill').forEach(bar => {
    const targetWidth = bar.style.width;
    bar.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        bar.style.width = targetWidth;
      });
    });
  });

  // ── STAGGER CARD ANIMATIONS ──
  document.querySelectorAll('.card').forEach((card, i) => {
    card.style.animationDelay = `${i * 60}ms`;
  });

  // ── KPI VALUE COUNT-UP (dashboard) ──
  document.querySelectorAll('.kpi-val').forEach(el => {
    const raw  = el.textContent.trim();
    const num  = parseFloat(raw.replace('%', ''));
    const unit = raw.includes('%') ? '%' : '';
    if (isNaN(num)) return;

    let start     = 0;
    const end     = num;
    const decimal = raw.includes('.') ? raw.split('.')[1]?.length || 0 : 0;
    const dur     = 900;
    const step    = 16;
    const inc     = (end / dur) * step;

    const timer = setInterval(() => {
      start += inc;
      if (start >= end) {
        start = end;
        clearInterval(timer);
      }
      el.textContent = start.toFixed(decimal) + unit;
    }, step);
  });

  // ── DTI QUICK CALCULATOR (index form) ──
  function updateDTI() {
    const income    = parseFloat(document.getElementById('income')?.value)    || 0;
    const loanAmt   = parseFloat(document.getElementById('loan_amount')?.value) || 0;
    const dtiField  = document.getElementById('dti');
    if (!dtiField || income <= 0) return;

    // Simple approximation: monthly loan payment / monthly income
    const monthlyPayment = loanAmt / 36;
    const monthlyIncome  = income  / 12;
    const dtiEst         = (monthlyPayment / monthlyIncome * 100).toFixed(2);

    // Only suggest if user hasn't manually changed DTI
    if (!dtiField.dataset.userEdited) {
      dtiField.value = dtiEst;
    }
  }

  const incomeField    = document.getElementById('income');
  const loanAmtField   = document.getElementById('loan_amount');
  const dtiField       = document.getElementById('dti');

  if (incomeField)  incomeField.addEventListener('change', updateDTI);
  if (loanAmtField) loanAmtField.addEventListener('change', updateDTI);
  if (dtiField)     dtiField.addEventListener('input', () => {
    dtiField.dataset.userEdited = '1';
  });

  // ── RISK PREVIEW BADGE ──
  function updateRiskPreview() {
    const fico = parseFloat(document.getElementById('credit_score')?.value) || 685;
    const dti  = parseFloat(document.getElementById('dti')?.value)           || 18;
    const rate = parseFloat(document.getElementById('int_rate')?.value)      || 11;

    let score = 0;
    if (fico < 650)  score += 2;
    else if (fico < 700) score += 1;
    if (dti > 35)   score += 2;
    else if (dti > 25) score += 1;
    if (rate > 18)  score += 2;
    else if (rate > 14) score += 1;

    let badge = document.getElementById('riskPreview');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'riskPreview';
      badge.style.cssText = `
        display:inline-block; font-size:0.72rem; font-weight:500;
        padding:3px 10px; border-radius:20px; margin-left:10px;
        transition: all 0.2s;
      `;
      const hdr = document.querySelector('.card-header h2');
      if (hdr) hdr.appendChild(badge);
    }

    if (score >= 4) {
      badge.textContent = '⚠ High Risk Profile';
      badge.style.background = 'var(--danger-light)';
      badge.style.color = 'var(--danger)';
    } else if (score >= 2) {
      badge.textContent = '~ Moderate Risk';
      badge.style.background = 'var(--warning-light)';
      badge.style.color = 'var(--warning)';
    } else {
      badge.textContent = '✓ Low Risk Profile';
      badge.style.background = 'var(--success-light)';
      badge.style.color = 'var(--success)';
    }
  }

  ['credit_score', 'dti', 'int_rate'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateRiskPreview);
  });
  updateRiskPreview(); // run on load

  // ── TOOLTIP FOR FORM LABELS ──
  document.querySelectorAll('.form-group label').forEach(label => {
    label.style.cursor = 'default';
  });

  console.log('[Loan Predictor] Script loaded ✓');
});
