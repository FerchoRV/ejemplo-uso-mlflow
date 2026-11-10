# credit-risk @ challenger

- URI: `models:/credit-risk@challenger`
- Versión: 1
- Endpoint: `POST http://127.0.0.1:5001/invocations`

## Esquema de entrada

```json
[
  {
    "type": "double",
    "name": "person_age",
    "required": true
  },
  {
    "type": "double",
    "name": "person_income",
    "required": true
  },
  {
    "type": "long",
    "name": "person_home_ownership",
    "required": true
  },
  {
    "type": "double",
    "name": "person_emp_length",
    "required": true
  },
  {
    "type": "long",
    "name": "loan_intent",
    "required": true
  },
  {
    "type": "double",
    "name": "loan_amnt",
    "required": true
  },
  {
    "type": "double",
    "name": "loan_int_rate",
    "required": true
  },
  {
    "type": "double",
    "name": "loan_percent_income",
    "required": true
  }
]
```

## Esquema de salida

```json
[
  {
    "type": "long",
    "required": true
  }
]
```

## Petición

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income"], "data": [[-0.44220311260130146, -0.12484565079562059, 3.0, -0.4415669058833856, 3.0, -1.0521712618466563, 0.7503988094474996, -1.1235941837207022]]}}'
```

```json
{
  "dataframe_split": {
    "columns": [
      "person_age",
      "person_income",
      "person_home_ownership",
      "person_emp_length",
      "loan_intent",
      "loan_amnt",
      "loan_int_rate",
      "loan_percent_income"
    ],
    "data": [
      [
        -0.44220311260130146,
        -0.12484565079562059,
        3.0,
        -0.4415669058833856,
        3.0,
        -1.0521712618466563,
        0.7503988094474996,
        -1.1235941837207022
      ]
    ]
  }
}
```

## Respuesta

```json
{
  "predictions": [
    1
  ]
}
```
