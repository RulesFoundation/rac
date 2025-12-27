# REST API

## Base URL

```
https://api.rac.ai/v1
```

## Calculate

```http
POST /calculate
Content-Type: application/json

{
  "household": {...},
  "jurisdictions": ["us", "us-ca"],
  "year": 2024,
  "variables": ["income_tax", "eitc"]
}
```

Response:
```json
{
  "us": {
    "income_tax": 4235.00,
    "eitc": 1502.00
  },
  "us_ca": {
    "income_tax": 892.00
  }
}
```

## Trace

```http
POST /trace
Content-Type: application/json

{
  "household": {...},
  "variable": "us.eitc"
}
```

Returns the calculation dependency tree with intermediate values.

## Parameters

```http
GET /parameters/us/irc/.../§32/(b)/(1)/credit_percentage
```

Returns parameter values across time.
