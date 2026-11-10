# Musk corpus expansion B: authored material and institutional events

This batch contains **12 additional primary sources and 71 short evidence records**. It is staged separately from the active persona corpus for review and integration. The records cover 2003–2019, use 76–136 words of evidence per source, and preserve the difference between direct testimony, an event host's report, and a coauthored corporate letter.

## Current source inventory

| Source | Date used | Material actually reviewed | Evidence limits |
| --- | --- | --- | --- |
| [Stanford Entrepreneurial Thought Leaders](https://stvp.stanford.edu/av/career-development) | 2003-10-08 | Three official clip transcripts from one event | One source, six records; no full-event timestamp claims |
| [Hyperloop Alpha](https://www.tesla.com/sites/default/files/blog_images/hyperloop-alpha.pdf) | 2013-08-12 | First-person introduction, printed pages 1–4 | Six records; an unfinished proposal, not measured performance |
| [MIT AeroAstro centennial conversation](https://news.mit.edu/2014/aeroastro-centennial-symposium-1025) | 2014-10-25 | Event host's report of Musk's exchange with Jaime Peraire | Four records; explicitly an excerpted report, not a complete transcript |
| [Senate launch-program statement](https://www.govinfo.gov/content/pkg/CHRG-113shrg49104594/html/CHRG-113shrg49104594.htm) | 2014-03-05 | Musk's prepared testimony | Five records; interested-party advocacy, not independently verified industry comparisons |
| [Tesla Company Update](https://www.sec.gov/Archives/edgar/data/1318605/000156459019000740/tsla-ex991_6.htm) | 2019-01-18 | Publicly filed employee letter signed Elon | Six records; statements and plans are anchored to the letter's historical date |
| [House hearing on NASA–Defense cooperation](https://www.govinfo.gov/content/pkg/CHRG-108hhrg92514/html/CHRG-108hhrg92514.htm) | 2004-03-18 | Oral statement plus separately labeled written answers | Seven records; written-answer date is unspecified |
| [Senate hearing on future launch vehicles](https://www.govinfo.gov/content/pkg/CHRG-108shrg20624/html/CHRG-108shrg20624.htm) | 2004-05-05 | Musk's oral statement | Six records; a different event from the March House hearing |
| [House hearing on commercial space](https://www.govinfo.gov/content/pkg/CHRG-109hhrg20541/html/CHRG-109hhrg20541.htm) | 2005-04-20 | Musk's opening statement in Panel II | Six records; preceding remarks by Burt Rutan excluded |
| [Senate hearing on advanced energy](https://www.govinfo.gov/content/pkg/CHRG-110shrg36077/html/CHRG-110shrg36077.htm) | 2007-03-07 | Musk's oral statement as Tesla chairman | Six records; attributed policy advocacy |
| [House hearing on commercial crew](https://www.govinfo.gov/content/pkg/CHRG-112hhrg70800/html/CHRG-112hhrg70800.htm) | 2011-10-26 | Musk's opening oral statement | Seven records; other witnesses' statements excluded |
| [Update on Taking Tesla Private](https://files.cand.uscourts.gov/files/18-cv-04865-EMC%20In%20re%20Tesla%20Inc.%20Securities%20Litigation/Trial%20Exhibit%200053%20-%208.13.18%20Blog%20Post.pdf) | 2018-08-13 | Original blog preserved as public court exhibit 53 | Six records; captures his explanation, not proof of financing claims |
| [2013 year-end shareholder letter](https://www.sec.gov/Archives/edgar/data/1318605/000119312514058778/d676319dex991.htm) | 2014-02-19 | SEC-filed letter co-signed by Musk and Deepak Ahuja | Six records; all evidence explicitly preserves corporate coauthorship |

The Stanford components are [Career Development](https://stvp.stanford.edu/node/18391/printable/print), [To MBA or Not?](https://stvp.stanford.edu/node/18461/printable/print), and [Decreasing Costs in Space Travel](https://stvp.stanford.edu/node/18496/printable/print). Their separate clip timelines are not combined into fabricated timestamps. The first clip's observed video link is [IWmXx04mf5Q](https://www.youtube.com/watch?v=IWmXx04mf5Q).

The Senate date is also confirmed by the [official hearing schedule](https://www.appropriations.senate.gov/news/minority/schedule-for-the-week-of-march-3-2014). The Hyperloop PDF does not print a release date; its August 12 release is documented by the [contemporaneous AllThingsD announcement](https://allthingsd.com/20130812/hyperloop-elon-musks-really-rapid-transit-system/). That separate date provenance is retained in `coverage.json`.

For hearings, the source date means the day the testimony was publicly delivered. It does not pretend to be the later digitization or republication date. The May 2004 Senate volume displays a 2016 publication year, and the October 2011 House volume displays 2012. The post-hearing answers in the March 2004 record are labeled as later written material with an unspecified submission date. The [2014 Senate GPO record](https://www.govinfo.gov/content/pkg/CHRG-113shrg49104594/html/CHRG-113shrg49104594.htm) also preserves the statement and provides an accessible alternative when the committee PDF returns 403.

## What this material can support

The curated passages add earlier career decisions, learning approaches, engineering tradeoffs, responses to resource constraints, explicit uncertainty, public procurement arguments, and employee communications. Each record preserves a source and a section-level locator. It describes a statement made in a particular setting; it is not a claim to know Musk's private reasoning or a timeless personality label.

Coverage is deliberately bounded: six sources are congressional events, two are university events, and four are authored or coauthored documents. This creates a strong emphasis on technical and business advocacy. The material cannot establish the frequency of a habit across his entire public life. The MIT host report offers selected attributed remarks, while the coauthored letter supports a corporate position he signed; neither should be treated as a complete individual interview.

All `behavioral_observation` fields remain `null`. Text does not establish facial movement, gesture frequency, gaze, prosody, or timing. No video or voice sample is distributed with this batch. The sources cannot justify saying a generated animation was observed in Musk's behavior.

## Curation and reproducibility

Each source contributes fewer than 180 words of evidence text, with no copied quotation passages. Whole articles and transcripts are not included. Multiple clips from one event count as one source. Other speakers' statements are excluded. Forecasts remain historical proposals; the curator does not promote them to facts about later outcomes.

The 71 curated paraphrases explicitly carry `review_status: "reviewed"` and `confidence: "source_grounded"` to record this source-curation pass. These labels do not claim automatic verification or an independent factual audit. New evidence with an omitted review flag defaults to draft and is excluded from retrieval; raw acquisitions and imported transcripts require their own attribution and content review before explicit promotion.

Validation against the repository's actual `Source` and `Evidence` models passed for all records. IDs are unique, source references resolve, and there is no source-ID or canonical-URL overlap with the original seven-source corpus or expansion A. Each source contributes 4–7 records; the evidence text totals 1,357 whitespace-separated words. These are structural and curation checks, not a persona-quality score or independent factual audit.

`data/expansion/batch-b/index.json` preserves the source and evidence IDs for this contribution. The canonical `data/personas/elon-musk/` corpus stores its metadata and paraphrases once; `coverage.json` retains access status, component URLs, speaker attribution, date provenance, and timing limits. The batch is integrated into the active corpus.

Some historical Tesla blog URLs now redirect to an index, and the linked Caltech 2012 transcript returned 404. A short Caltech host report and a brief SEC-filed September 2018 employee message were available but were not padded into extra sources. The official USC 2014 video was found, but the retrieved transcript text came from third parties and is not used here. These discoveries are logged separately from acquired sources. Senate PDF access succeeded during research and later intermittently returned 403; the same selected content was rechecked in the official GPO transcript.
