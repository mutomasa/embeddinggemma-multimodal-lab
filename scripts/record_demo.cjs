// Records a walkthrough of the Streamlit demo (README「デモ」).
// Usage: node scripts/record_demo.cjs <out_dir>   (Streamlit must be running on BASE_URL)
const { chromium } = require(process.env.PW_CORE || 'playwright');

const BASE = process.env.BASE_URL || 'http://localhost:8501/';
const OUT_DIR = process.argv[2] || 'artifacts/demo_video';

async function main() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: OUT_DIR, size: { width: 1280, height: 800 } },
  });
  const p = await ctx.newPage();

  const idle = async (extra = 1500) => {
    await p.waitForTimeout(800);
    await p.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 300000 }).catch(() => {});
    await p.waitForTimeout(extra);
  };
  const scroll = async (dy, steps = 8) => {
    for (let i = 0; i < steps; i++) { await p.mouse.wheel(0, dy / steps); await p.waitForTimeout(120); }
    await p.waitForTimeout(1200);
  };
  const query = async (box, text) => {
    await box.click();
    await box.fill('');
    await box.pressSequentially(text, { delay: 60 });
    await box.press('Enter');
    await idle(2500);
  };

  try {
    // 1. Search: 日英クエリ → 次元 768 → 128
    await p.goto(BASE + 'Search'); await idle(2500);
    await p.mouse.move(640, 400);
    await query(p.getByRole('textbox', { name: 'クエリ（日本語・英語）' }), 'ロボットが物体を落とす');
    await scroll(400); await scroll(-400);
    await p.getByRole('radiogroup', { name: '埋め込み次元' }).getByText('128', { exact: true }).click(); await idle(3000);
    console.log('search ok');

    // 2. Multimodal Search: テキスト → 動画
    await p.goto(BASE + 'Multimodal_Search'); await idle(2000);
    await p.getByRole('tab', { name: 'テキスト → 動画' }).click(); await idle(1500);
    await query(p.getByRole('tabpanel', { name: 'テキスト → 動画' }).getByRole('textbox').first(), 'the robot drops the block');
    await scroll(500); await scroll(500); await scroll(-1000);
    console.log('multimodal ok');

    // 3. Embedding Analysis: UMAP → 類似度行列
    await p.goto(BASE + 'Embedding_Analysis'); await idle(3000);
    await p.getByRole('radiogroup', { name: '色分け' }).getByText('modality').click(); await idle(2500);
    await p.getByRole('tab', { name: '類似度行列' }).click(); await idle(3500);
    await scroll(400); await scroll(-400);
    console.log('analysis ok');

    // 4. Evaluation: 4 次元比較
    await p.goto(BASE + 'Evaluation'); await idle(1500);
    await p.getByRole('button', { name: '評価を実行' }).click(); await idle(3000);
    await scroll(600); await scroll(600); await scroll(600); await scroll(-1800);
    console.log('evaluation ok');

    // 5. Research Demo: 落下クエリ → 人手ラベル照合 → 4 次元評価
    await p.goto(BASE + 'Research_Demo'); await idle(2500);
    await p.getByRole('combobox', { name: 'クエリ（プリセット）' }).click(); await p.waitForTimeout(1200);
    await p.getByRole('option').filter({ hasText: /^r05/ }).click(); await idle(3000);
    await scroll(500); await scroll(500); await scroll(500);
    await p.getByRole('button', { name: '評価を実行' }).click(); await idle(3000);
    await scroll(600); await scroll(600);
    await p.waitForTimeout(2000);
    console.log('research ok');
  } finally {
    const video = p.video();
    await ctx.close();
    console.log('video:', await video.path());
    await browser.close();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
