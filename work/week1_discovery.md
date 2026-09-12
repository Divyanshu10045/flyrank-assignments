# Discovery: avg_position vs CTR

I tested the correlation between average search position and click-through rate using `df['avg_position'].corr(df['ctr'])`. The result was **-0.073**, which is a very weak negative correlation. The direction matches the general SEO intuition (better/lower position tends to associate with slightly higher CTR), but the magnitude is small enough that position alone explains very little of the variation in CTR in this dataset.

This is an observed, directional pattern in the anonymized sample, not a claim about how Google's algorithm works, and not a causal relationship. It suggests CTR here is likely driven more by other factors (e.g. content type, query intent, or branding) than by position alone, which would be worth checking in a future discovery.
