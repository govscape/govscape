<script>
  import { searchStore, searchActions } from '$lib/stores/search';

  // Follows the store, e.g. when a search link sets the weights.
  $: weights = $searchStore.hybridWeights || { textual: 0.33, visual: 0.33, keyword: 0.34 };

  function updateWeight(key, value) {
    searchActions.setHybridWeights({ [key]: Number(value) });
  }
</script>

<div class="hybrid-sliders">
  <div class="slider-row">
    <label for="textual">Textual</label>
    <input id="textual" type="range" min="0" max="1" step="0.01" value={weights.textual} on:input={(e) => updateWeight('textual', e.target.value)} />
    <div class="value">{Math.round(weights.textual * 100)}%</div>
  </div>
  <div class="slider-row">
    <label for="visual">Visual</label>
    <input id="visual" type="range" min="0" max="1" step="0.01" value={weights.visual} on:input={(e) => updateWeight('visual', e.target.value)} />
    <div class="value">{Math.round(weights.visual * 100)}%</div>
  </div>
  <div class="slider-row">
    <label for="keyword">Keyword</label>
    <input id="keyword" type="range" min="0" max="1" step="0.01" value={weights.keyword} on:input={(e) => updateWeight('keyword', e.target.value)} />
    <div class="value">{Math.round(weights.keyword * 100)}%</div>
  </div>
  <div class="note">Weights are sent as-is and normalized server-side.</div>
</div>

<style>
  .hybrid-sliders {
    margin-top: 12px;
    width: 100%;
    max-width: 600px;
    background: #fff;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0 1px 6px rgba(32,33,36,0.06);
  }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .slider-row label {
    width: 90px;
    font-size: 0.85rem;
    color: var(--text-color-secondary);
  }
  .slider-row input[type=range] {
    flex: 1 1 0;
  }
  .slider-row .value {
    width: 48px;
    text-align: right;
    font-size: 0.85rem;
    color: var(--text-color-primary);
  }
  .note {
    margin-top: 6px;
    font-size: 0.8rem;
    color: var(--text-color-secondary);
  }
</style>
