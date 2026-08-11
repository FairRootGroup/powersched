### prices_2023.csv

German day-ahead spot market electricity prices for 2023 (bidding zone
DE-LU), hourly resolution.

**Provenance:** originally sourced from
[Bundesnetzagentur / SMARD.de](https://www.smard.de/), republished by
[Energy-Charts](https://energy-charts.info/) (Fraunhofer ISE) under
CC BY 4.0 — see [LICENSE](LICENSE).

**Reproduction:** the file can be regenerated from the Energy-Charts API:

```bash
curl "https://api.energy-charts.info/price?bzn=DE-LU&start=2023-01-01&end=2023-12-31"
```

This returns JSON (`unix_seconds` + `price` arrays); `prices_2023.csv` is
the same data reformatted as `Date,Price` with `Date` in local time
(`Europe/Berlin`, i.e. UTC+01:00/+02:00 depending on DST).

**Known quirk:** 3 of the 8760 rows carry a mislabeled UTC offset in the
`Date` column (`2023-02-26T03:00+00:00` and `2023-09-29T03:00+03:00` /
`04:00+03:00`, which should read `+01:00` / `+02:00` respectively — neither
date is a DST transition in Germany). The hour ordering and price values
are correct; only those 3 offset strings are wrong. Confirmed by diffing
against the API response above (all 8760 price values match once the
offset typo is accounted for).
