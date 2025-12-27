# JavaScript API

## Installation

```bash
npm install @cosilico/engine @cosilico/us
```

## Basic Usage

```javascript
import { Simulation } from '@cosilico/engine';
import '@cosilico/us';

const sim = new Simulation({
  jurisdictions: ['us'],
  year: 2024
});

const household = {
  people: {
    adult: { age: 30, employment_income: 50000 }
  },
  tax_units: {
    tax_unit: { members: ['adult'], filing_status: 'single' }
  },
  households: {
    household: { members: ['adult'], state_name: 'CA' }
  }
};

const result = sim.calculate(household);
console.log(result.us.income_tax);
```

## Browser Usage

The WASM build runs entirely in-browser:

```html
<script type="module">
import { Simulation } from 'https://cdn.rac.ai/engine.js';

const sim = new Simulation({ jurisdictions: ['us'], year: 2024 });
// ...
</script>
```
