# Jensen Huang public-source acquisition — 2026-11-24

This batch broadens public evidence for the project across biography, decisions, technical explanations, university appearances, company announcements, and presentation history. Publisher pages were reviewed before evidence was written. Search-discovered links and inaccessible media remain separate from active evidence.

## Shards and built result

| File group | Source records in shard | Evidence records | Records with short original excerpts |
|---|---:|---:|---:|
| `biography-sources.json` / `biography-evidence.jsonl` | 21 | 48 | 9 |
| `judgment-sources.json` / `judgment-evidence.jsonl` | 18 | 53 | 17 |
| New evidence total | Source IDs may overlap existing sources | 101 | 26 |

The builder merges source IDs into the retained values-v2 baseline and excludes reserved source-event families. The resulting [public-v3 corpus](../../personas/jensen-huang-public-v3/README.md) has **46 sources, 39 event families, and 155 evidence records: 54 retained plus 101 new**. Source counts in the two acquisition shards are not an additional-source count.

Files have distinct roles:

- `*-sources.json` identifies publisher, URL, dates and date basis, event family, split, and provenance.
- `*-evidence.jsonl` contains bounded paraphrases, authored English/Korean search anchors, source locators, and explicitly separated decision/style annotations and short excerpts.
- `*-discovery.json` records unread or excluded leads, access failures, attribution conflicts, and reuse cautions.
- `judgment-heldout.json` contains a separate reserved evaluation protocol, with status `prepared_not_executed`.
- `capture-manifest.json` describes exact original document captures; `capture-receipt.json` records persistence and independent hash verification.

Review is agent review against the cited public publisher material, not human certification. Institutional biographies and corporate accounts are attributed as such; their publication does not make every statement an independently verified personal recollection. New search anchors and reasoning annotations are editorial work, not invented original Q&A.

## Original document capture and dates

A follow-up acquisition captured **8 of 9 requested documents** using their registered public URLs: **7 HTML files and 1 PDF**, with **1,848,016 original response-body bytes** in total. No login, access-control workaround, audio, or video download was used. Accepted files passed substantive title/topic checks; exact byte lengths and SHA-256 values are retained in [capture-manifest.json](capture-manifest.json).

The archive was uploaded and independently materialized, and the downloaded copy's SHA-256 matched the original. The authoritative persistence details are in [capture-receipt.json](capture-receipt.json):

- Archive: `jensen-source-capture-2026-11-24.zip`.
- Archive size: 848,160 bytes.
- Archive SHA-256: `d8b88a2f9a9a4ac8fe43f2c9cc47839a02202487642986567f1dc3efbe65ae09`.
- Capture interval for accepted files: 2026-11-24 11:18:19–11:18:41 UTC; each exact timestamp is in the manifest.

Raw documents remain outside Git. Git stores the provenance manifest, receipt, and bounded runtime notes. Original documents in the archive are verification material; their presence does not mean a complete Q&A dataset has been prepared or that original speech timing was checked. Statements in the initial discovery logs about material not acquired describe that initial research phase; the later capture manifest identifies the exact subsequent acquisitions.

A capture timestamp proves observed availability at that instant. It does not establish original publication, a recording date, or earlier availability. In particular, the new Milken capture proves November 24 availability, not a November 21 capture. For the undated captured pages, `publication_date_verified` remains false and `publication_date_basis` is `observed_availability`.

`stanford-etl-2003-time` returned **HTTP 404** at its exact registered PDF URL. No alternate document was silently substituted. Its three records remain in the reviewable corpus but are excluded by the recorded snapshot's general temporal selection. Event dates, award years, dedication dates, website footers, and historical milestone dates are not substituted for publication dates. The separate OSU article `jhb-osu-hard-work-2013` has a verified publisher byline date of **2013-02-25**.

## Instagram and other discovery-only material

Instagram was explicitly checked. NVIDIA's company profile and three specific post/reel URLs are recorded in [biography-discovery.json](biography-discovery.json). The profile returned HTTP 429 and the three post/reel pages could not be read through the available access path. Their captions, speakers, dates, and media were not acquired or indexed. Aggregator snippets and a public-forum link establish discovery only.

One NVIDIA filing hosted by the SEC, `jhb-sec-social-2026`, supplies indexed evidence of official channel identity. This distinguishes the company's Instagram account from any supposed personal Jensen Instagram account. The filing's channel listing does not authenticate arbitrary posts or screenshots. Jensen's listed X account was also inaccessible to direct retrieval, and its posts were not indexed.

HP, OSU commencement, and HKUST original-video leads remain discovery metadata unless a separate readable publisher source supports an indexed note. No Priors 2024 and an a16z multi-speaker episode remain outside active evidence because their metadata/synopsis did not establish verified original dialogue or individual attribution.

## Conflicts, source events, and reserved evaluation

The discovery logs retain material exclusions: differing birthplace/arrival accounts were not collapsed into false certainty; NTU's publication and ceremony dates remain separate; Caltech's erroneous AlexNet chronology was excluded; and AI-generated Toy Jensen utterances are not treated as the real speaker's experiences or style.

The NVIDIA timeline records CUDA's **2006 unveiling**, a milestone distinct from later releases/adoption, and AlexNet in **2012**. Its corporate provenance remains explicit. Multiple university pages, clips, or company reports describing one event share that event family rather than counting as independent observations.

The [older source-family register](../jensen-huang/source-families.json) reserves No Priors 2023 and WEF Davos 2026. Six No Priors 2023 records inherited by the old values-v2 experiment are removed from the new active corpus; the old files remain intact. WEF is retained only as an excluded discovery lead. The NBA Technology Summit 2025 event is separately reserved in [judgment-heldout.json](judgment-heldout.json), excluded from the active corpus and prompts, and **prepared but not evaluated**. Its proposed blind comparison separates factual support, decision criteria, discourse/style, and completion/latency; voice comparison would use the same answer text.

## Complete built source catalog

This table lists the **46 source records in the built corpus**, including retained baseline sources. `Source material` describes the publisher document, not the runtime evidence format. An observed date is an availability bound with the original publication date unknown. The uncaptured provisional date is not a claim of publication. Event grouping and precise date/provenance fields in [sources.json](../../personas/jensen-huang-public-v3/sources.json) and [split-manifest.json](../../personas/jensen-huang-public-v3/split-manifest.json) are authoritative.

| Source ID | Publisher and source | Stored date and basis | Source material | Event family |
|---|---|---|---|---|
| `acquired-2023` | Acquired: [NVIDIA CEO Jensen Huang](https://www.acquired.fm/episodes/jensen-huang#transcript) | 2023-10-16 | publisher interview transcript | `acquired-2023` |
| `citadel-sequoia-2025` | Citadel Securities: [Nvidia’s Jensen Huang on AI & the Next Frontier of Growth](https://www.citadelsecurities.com/wp-content/uploads/sites/2/2025/11/Nvidias-Jensen-Huang-on-AI-the-Next-Frontier-of-Growth.pdf) | 2025-11-14 | publisher interview transcript | `citadel-sequoia-2025` |
| `cnbc-jensen-2026-03-17` | CNBC / VERSANT Press Room: [CNBC Exclusive: Transcript: Nvidia Founder & CEO Jensen Huang Speaks with CNBC’s Jim Cramer](https://pressroom.versantmedia.com/cnbc/press-releases/cnbc-exclusive-transcript-nvidia-founder-ceo-jensen-huang-speaks-cnbcs-jim) | 2026-03-17 | publisher interview transcript | `nvidia-gtc-2026` |
| `csis-2025` | CSIS: [NVIDIA’s Jensen Huang on Securing American Leadership on AI](https://www.csis.org/analysis/nvidias-jensen-huang-securing-american-leadership-ai) | 2025-12-03 | publisher interview transcript | `csis-2025` |
| `dwarkesh-2026` | Dwarkesh Podcast: [Jensen Huang on TPU competition and NVIDIA’s supply chain](https://www.dwarkesh.com/p/jensen-huang) | 2026-04-15 | publisher interview transcript | `dwarkesh-2026` |
| `jhb-caltech-2024` | California Institute of Technology: [Caltech Honors Graduates at 130th Commencement Ceremony](https://www.caltech.edu/about/news/caltech-commencement-2024) | 2024-06-14 | university commencement report | `caltech-commencement-2024` |
| `jhb-hkust-2024` | Hong Kong University of Science and Technology: [From Quips to Chips: Dr. Jensen Huang Spots the World of AI](https://hkust.edu.hk/news/quips-chips-dr-jensen-huang-spots-world-ai) | 2024-12-07 | university interview report | `hkust-huang-shum-2024` |
| `jhb-hoover-only-in-america` | Hoover Institution: [Only In America: Jensen Huang on Vision, Risk, and the GPU](https://www.hoover.org/plus/research/jensen-huang-vision-risk-and-gpu) | 2026-11-24 observed; publication unknown | publisher interview transcript | `hoover-only-in-america-huang` |
| `jhb-ieee-founders-2020` | IEEE: [2020 IEEE Honors Ceremony](https://ieeetv.ieee.org/event/ieee-honors-2020) | 2020-05-15 | award body event | `ieee-honors-2020` |
| `jhb-kitchen-keynote-2020` | NVIDIA Technical Blog: [Watch the NVIDIA GTC 2020 Keynote](https://developer.nvidia.com/blog/watch-the-nvidia-gtc-2020-keynote/) | 2020-05-14 | publisher keynote announcement | `nvidia-gtc-2020` |
| `jhb-ntu-ceremony-2023` | National Taiwan University: [NTU Commencement 2023](https://www.ntu.edu.tw/english/spotlight/2023/2164_20230529.html) | 2023-05-29 | university commencement report | `ntu-commencement-2023` |
| `jhb-nvidia-dennys-2023` | NVIDIA Blog: [Jensen Huang Returns to Denny’s Where NVIDIA Launched a Trillion-Dollar Vision](https://blogs.nvidia.com/blog/nvidia-dennys-trillion/) | 2023-09-26 | publisher event report | `nvidia-dennys-2023` |
| `jhb-nvidia-history` | NVIDIA: [Our History: Innovations Over the Years](https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/) | 2026-11-24 observed; publication unknown | company historical timeline | `nvidia-corporate-history` |
| `jhb-nvidia-ntu-2023` | NVIDIA Blog: [NVIDIA CEO Tells NTU Grads to Run, Not Walk — But Be Prepared to Stumble](https://blogs.nvidia.com/blog/huang-ntu-commencement/) | 2023-05-26 | publisher commencement report | `ntu-commencement-2023` |
| `jhb-nvidia-official-bio` | NVIDIA Newsroom: [Jensen Huang](https://nvidianews.nvidia.com/bios/jensen-huang) | 2026-11-24 observed; publication unknown | company executive biography | `nvidia-official-biography` |
| `jhb-oneida-hall-2019` | Oneida Baptist Institute: [Jen-Hsun Huang Hall Construction](https://www.oneidaschool.org/about/huanghall.cfm) | 2026-11-24 observed; publication unknown | school facility report | `oneida-huang-hall-2019` |
| `jhb-osu-gift-2022` | Oregon State University: [$50 Million Gift by NVIDIA Founder and Spouse Helps Launch Oregon State University Research Center](https://news.oregonstate.edu/news/50-million-gift-nvidia-founder-and-spouse-helps-launch-oregon-state-university-research-center) | 2022-10-14 | university gift announcement | `osu-huang-complex-gift-2022` |
| `jhb-osu-hall-2013` | Oregon State University College of Engineering: [Jen-Hsun Huang: Engineering Hall of Fame - 2013](https://engineering.oregonstate.edu/alumni-partners/oregon-stater-awards/searchable-awards-database/jen-hsun-huang-engineering-hall) | 2026-11-24 observed; publication unknown | university biography | `osu-hall-of-fame-2013` |
| `jhb-osu-hard-work-2013` | Oregon State University College of Engineering: [Doing important, hard work you love](https://engineering.oregonstate.edu/all-stories/doing-important-hard-work-you-love) | 2013-02-25 | university event report | `osu-hall-of-fame-2013` |
| `jhb-sec-social-2026` | NVIDIA filing hosted by U.S. SEC: [NVIDIA Form 10-Q for the quarter ended July 26, 2026](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm) | 2026-08-26 document signature | company regulatory filing | `nvidia-fy2027-q2-filing` |
| `jhb-sia-noyce-2021` | Semiconductor Industry Association: [NVIDIA Founder and CEO Jensen Huang to Receive Semiconductor Industry’s Top Honor](https://www.semiconductors.org/nvidia-founder-and-ceo-jensen-huang-to-receive-semiconductor-industrys-top-honor/) | 2021-08-12 | award body announcement | `sia-noyce-huang-2021` |
| `jhb-siepr-2024` | Stanford Institute for Economic Policy Research: [Nvidia’s Jensen Huang: The incredible future of AI](https://siepr.stanford.edu/news/nvidias-jensen-huang-incredible-future-ai) | 2024-03-07 | university interview report | `stanford-siepr-2024` |
| `jhb-stanford-pillars` | Stanford University School of Engineering: [Pillars](https://engineering.stanford.edu/about/visit/pillars) | 2026-11-24 observed; publication unknown | university historical exhibit | `stanford-huang-engineering-history` |
| `jhb-toy-jensen-2021` | NVIDIA Blog: [How Omniverse Wove a Real CEO — and His Toy Counterpart — Together With Stunning Demos at GTC](https://blogs.nvidia.com/blog/how-omniverse-keynote/) | 2021-12-21 | company production report | `nvidia-gtc-fall-2021` |
| `jhb-virtual-jensen-2021` | NVIDIA Blog: [From Our Kitchen to Yours: NVIDIA Omniverse Changes the Way Industries Collaborate](https://blogs.nvidia.com/blog/omniverse-making-of-gtc/) | 2021-08-11 | company production report | `nvidia-gtc-spring-2021` |
| `lex-494` | Lex Fridman Podcast: [Lex Fridman Podcast #494: Jensen Huang](https://lexfridman.com/jensen-huang-transcript/) | 2026-03-23 | publisher interview transcript | `lex-494` |
| `milken-2026` | Milken Institute: [Leading in the Age of AI: A Conversation with NVIDIA CEO Jensen Huang](https://milkeninstitute.org/sites/default/files/2026-05/LeadingAgeAIAConversationNVIDIACEOJensenHuang_Transcript_GC26.pdf) | 2026-11-24 observed; publication unknown | publisher transcript | `milken-2026-05-04` |
| `nvidia-blackwell-ultra-2025` | NVIDIA Newsroom: [NVIDIA Blackwell Ultra AI Factory Platform Paves Way for Age of AI Reasoning](https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning) | 2025-03-18 | company press release | `nvidia-gtc-2025` |
| `nvidia-ces-2025-report` | NVIDIA Blog: [CES 2025: AI Advancing at Incredible Pace, NVIDIA CEO Says](https://blogs.nvidia.com/blog/ces-2025-jensen-huang/) | 2025-01-06 | publisher keynote report | `nvidia-ces-2025` |
| `nvidia-computex-2024` | NVIDIA Blog: [Accelerate Everything: NVIDIA CEO Says Ahead of COMPUTEX](https://blogs.nvidia.com/blog/computex-2024-jensen-huang/) | 2024-06-02 | publisher keynote report | `nvidia-computex-2024` |
| `nvidia-computex-2025` | NVIDIA Blog: [NVIDIA CEO Envisions AI Infrastructure Industry Worth Trillions of Dollars](https://blogs.nvidia.com/blog/computex-2025-jensen-huang/) | 2025-05-18 | publisher keynote report | `nvidia-computex-2025` |
| `nvidia-gtc-2024-report` | NVIDIA Blog: [GTC Wrap-Up: We Created a Processor for the Generative AI Era, NVIDIA CEO Says](https://blogs.nvidia.com/blog/2024-gtc-keynote/) | 2024-03-18 | publisher keynote report | `nvidia-gtc-2024` |
| `nvidia-gtc-2025-live-report` | NVIDIA Blog: [GTC 2025 – Announcements and Live Updates](https://blogs.nvidia.com/blog/nvidia-keynote-at-gtc-2025-ai-news-live-updates/) | 2025-03-25 | publisher keynote report | `nvidia-gtc-2025` |
| `sequoia-crucible-nvidia-2023` | Sequoia Capital: [Nvidia: An Overnight Success Story 30 Years in the Making](https://sequoiacap.com/podcast/crucible-moments-nvidia) | 2023-11-30 | publisher interview transcript | `sequoia-crucible-nvidia-2023` |
| `siggraph-meta-2024` | NVIDIA Blog: [Creators to Have Personalized AI Assistants, Meta CEO Mark Zuckerberg Tells NVIDIA CEO Jensen Huang](https://blogs.nvidia.com/blog/zuckerberg-huang/) | 2024-07-29 | publisher interview report | `siggraph-meta-2024` |
| `stanford-etl-2003-culture` | Stanford eCorner / STVP: [The Power of Corporate Culture](https://stvp.stanford.edu/node/19126/printable/print) | 2026-11-24 observed; publication unknown | publisher interview transcript | `stanford-etl-2003-01-29` |
| `stanford-etl-2003-time` | Stanford eCorner / STVP: [How a CEO Spends Time](https://stvp.stanford.edu/wp-content/uploads/sites/3/2024/09/how-a-ceo-spends-time-transcript.pdf) | 2026-11-24 provisional availability bound; publication unknown | publisher interview transcript | `stanford-etl-2003-01-29` |
| `stanford-etl-2009-allocation` | Stanford STVP: [Setting Prices, Evaluating Opportunities](https://stvp.stanford.edu/node/12966/printable/print) | 2026-11-21 observed; publication unknown | publisher clip transcript | `stanford-etl-2009-04-08` |
| `stanford-etl-2009-founding` | Stanford STVP: [The First Six Months of NVIDIA](https://stvp.stanford.edu/node/12976/printable/print) | 2026-11-21 observed; publication unknown | publisher clip transcript | `stanford-etl-2009-04-08` |
| `stanford-etl-2009-leaders` | Stanford STVP: [Cultivating Next-Generation Leaders](https://stvp.stanford.edu/node/12986/printable/print) | 2026-11-21 observed; publication unknown | publisher clip transcript | `stanford-etl-2009-04-08` |
| `stanford-etl-2009-reinvention` | Stanford STVP: [The Challenge of Constant Rebuilding](https://stvp.stanford.edu/node/12981/printable/print) | 2026-11-21 observed; publication unknown | publisher clip transcript | `stanford-etl-2009-04-08` |
| `stanford-gsb-2024` | Stanford Graduate School of Business: [Jensen Huang on How to Use First-Principles Thinking to Drive Decisions](https://www.gsb.stanford.edu/insights/jensen-huang-how-use-first-principles-thinking-drive-decisions) | 2024-04-25 | publisher transcript | `stanford-gsb-2024` |
| `strange-loop-2025` | Strange Loop (Sana): [Leading the teams behind the world’s fastest AI](https://sanalabs.com/strange-loop/jensen-huang) | 2025-06-26 | publisher interview transcript | `strange-loop-2025` |
| `stripe-sessions-2024` | Stripe Sessions: [A conversation with NVIDIA’s Jensen Huang](https://stripe.com/gb/sessions/2024/a-conversation-with-nvidias-jensen-huang) | 2024-05-21 | publisher interview transcript | `stripe-sessions-2024` |
| `world-governments-summit-2024` | NVIDIA Blog: [NVIDIA CEO: Every Country Needs AI](https://blogs.nvidia.com/blog/world-governments-summit/) | 2024-02-12 | publisher interview report | `world-governments-summit-2024` |
| `yc-startup-school-2026` | Y Combinator: [Jensen Huang: The Mindset That Built NVIDIA](https://www.ycombinator.com/library/Tq-jensen-huang-the-mindset-that-built-nvidia) | 2026-07-26 | publisher interview transcript | `yc-startup-school-2026` |
