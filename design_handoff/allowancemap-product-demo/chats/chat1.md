# Product Demo Animation

_Started 2026-04-24 17:41 UTC_

---

## User

https://claude.ai/design/p/7cc19415-7761-4195-9d3b-77f7ce98631d
We created an animation explaining the technical aspects of the app. We need to create something similar but for a regular audience and socials as a product demo and showcase its capabilities. Highlight the Fraud investigation agent as well on top of the main features. Keep it slick, fast and flowing.

## Assistant

I'll take a look at the existing animation to understand the style and approach, then plan the product demo version.

_[tool: list_files]_

_[tool: list_files]_

_[tool: read_file]_

_[tool: read_file]_

I'll continue exploring the existing animation structure and then build a product-demo version.

_[tool: read_file]_

_[tool: read_file]_

_[tool: read_file]_

I have a strong understanding of the source animation. Let me plan the product-demo version aimed at regular audiences + socials, then build it.

The new animation will:
- Be shorter, punchier, vertical-friendly thinking but in 1920×1080 (can be cropped)
- Skip technical depth (no teacher-forcing bug, no R² race, no pipeline internals)
- Focus on consumer value: "How much will this cost?" → answer; Forecast; **Fraud Investigation agent**
- Flow: Hook → The Question → Instant Answer (Cost Estimator) → Forecast Explorer → Fraud Agent (Claude) → CTA

Let me set up the project.

_[tool: copy_files]_

_[tool: list_files]_

_[tool: snip]_

_[tool: list_files]_

_[tool: list_files]_

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

I have enough context now to build the product demo. Let me create the scenes — I'll structure this as a punchy, consumer-facing demo with 7 scenes totaling ~40 seconds, focused on product capabilities, not technical internals.

Plan:
1. **Open** (0–2.5s): Bold hook — "How much will Medicare cost?"
2. **Estimator** (2.5–10s): Live UI mockup filling in, cost answer drops
3. **OOP Distribution** (10–15s): Patient-specific P10/P50/P90 
4. **Forecast** (15–22s): 2024–2026 chart drawing in
5. **Fraud Agent** (22–32s): Claude investigation brief — the standout feature
6. **Feature Grid** (32–37s): Quick recap of 3 capabilities
7. **CTA** (37–42s): AllowanceMap + URL, slick outro

Let me build the scene files:

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: done]_

_[tool: snip]_

_[tool: fork_verifier_agent]_

Built a ~43s product demo animation aimed at a general audience + socials — punchier and less technical than the original:

**Flow (7 scenes):**
1. **Hook** — "How much will Medicare *really* cost?"
2. **Cost Estimator** — form fills itself in, answer drops in real-time ($87.24)
3. **OOP Distribution** — P10/P50/P90 bell curve, personalized to the patient
4. **Forecast Explorer** — historical line + dashed 2024–2026 forecast with P10/P90 confidence band
5. **Fraud Investigation Agent** (the standout) — provider grid scans, one flags red, Claude-generated brief slides in with risk score, finding, rule checks, and analyst actions
6. **Feature recap** — three cards: Estimate · Project · Investigate
7. **CTA outro** — dark hero, wordmark, URL, "Powered by Claude"

Fully on-brand with the AllowanceMap palette (teal/green/amber, warm off-white), Inter + IBM Plex Mono, the signature 4px left-border result cards, and the risk red reserved for the fraud scene only.

## User

On the fifth section- the fraud detection agent section- extend it to 2 more seconds padding the initial part. The explanation of what the agent is and will do is too quick

## Assistant

_[tool: snip]_

I'll extend the Fraud scene by 2s and shift the pre-brief timing so the scanning/setup phase lasts longer before the brief slides in.

_[tool: snip]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: str_replace_edit]_

_[tool: snip]_

_[tool: done]_

_[tool: snip]_

Done — the Fraud scene now runs 12s (was 10s). The pre-brief hook phase is now ~3.8s (was ~1.8s), giving room to read "1.26M providers. One AI." while the provider grid scans at a slower pace. Brief slides in at 3.8s, rules reveal through 6s, actions highlight at 6.6s.

