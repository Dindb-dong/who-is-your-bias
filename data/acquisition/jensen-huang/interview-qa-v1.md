# Jensen Huang original interview Q&A index

Reviewed on 2026-11-12. This first collection indexes 16 exchanges across
four independently published English-language events. Each row summarizes an
actual interviewer prompt and Huang's response from the linked publisher
transcript. Use the locator to read the original turns in context.

See the [second source collection](source-catalog-v2.md) for the expanded event
list and additional Q&A.

## Sources and dates

| Event ID | Publisher and interviewer | Publication date | Event or recording date | Original material |
| --- | --- | --- | --- | --- |
| `lex-494` | Lex Fridman Podcast, Lex Fridman | 2026-03-23 (episode post) | Unknown | [Episode and date](https://lexfridman.com/jensen-huang/), [speaker-labeled transcript with linked timestamps](https://lexfridman.com/jensen-huang-transcript/) |
| `dwarkesh-2026` | Dwarkesh Podcast, Dwarkesh Patel | 2026-04-15 (episode post) | Unknown | [Episode, date and speaker-labeled transcript](https://www.dwarkesh.com/p/jensen-huang) |
| `acquired-2023` | Acquired, Ben Gilbert and David Rosenthal | 2023-10-16 03:09 UTC (distribution timestamp) | Unknown | [Episode and publisher transcript dialog](https://www.acquired.fm/episodes/jensen-huang#transcript), [release timestamp](https://podcasts.apple.com/us/podcast/nvidia-ceo-jensen-huang/id1050462261?i=1000631419166) |
| `stripe-sessions-2024` | Stripe Sessions, Patrick Collison | Unknown | 2024-04-23 to 2024-04-25 (conference dates; interview day unverified) | [Speaker-labeled interview transcript](https://stripe.com/gb/sessions/2024/a-conversation-with-nvidias-jensen-huang), [conference dates](https://stripe.com/blog/stripe-sessions-2024) |

The date of an episode post identifies when the publisher made it public.
Treat dated opinions and company claims below as statements made in the source.

## Lex Fridman Podcast #494

Times identify the starts of the relevant transcript blocks for the question
and answer. They are publisher locators.

| Exchange ID | Question → answer | Question in brief | Huang's answer in brief |
| --- | --- | --- | --- |
| `lex-494-01` | 00:00:42 → 00:01:11 | Why is designing at rack scale hard? | Splitting the workload across machines brings communication and coordination costs into play. He invokes Amdahl's law to explain why improving compute alone has limited effect. |
| `lex-494-02` | 00:16:45 → 00:17:14 | How does he build conviction for a high-risk company decision? | Curiosity starts the process. He works through why a future outcome should occur until he has enough conviction to endure the difficult period before it arrives. |
| `lex-494-03` | 01:09:48 → 01:10:19 | What makes TSMC successful? | He credits both process and packaging technology and TSMC's ability to coordinate shifting customer orders while honoring delivery commitments. |
| `lex-494-04` | 02:03:04–02:03:40 → 02:03:43 | How much should programmers specify when working with AI agents? | The appropriate detail depends on the problem and who will act on it. He uses company strategy as an example of setting enough direction for action while leaving room for others to improve it. |

## Dwarkesh Podcast

Times mark publisher chapter starts. The prompt description locates the
exchange within its chapter.

| Exchange ID | Chapter and position | Question in brief | Huang's answer in brief |
| --- | --- | --- | --- |
| `dwarkesh-2026-01` | 00:00:00, first supply-commitment follow-up | Is access to scarce components NVIDIA's main advantage? | He calls it one advantage and describes how downstream demand and alignment with suppliers give those suppliers a reason to expand capacity. |
| `dwarkesh-2026-02` | 00:16:25, opening exchange | What does the success of TPU-trained models mean for NVIDIA? | He frames NVIDIA as serving many accelerated computing workloads and points to software breadth and the ability of outside operators to run its systems. |
| `dwarkesh-2026-03` | 00:41:06, follow-up on becoming a cloud provider | Why support cloud companies instead of becoming a hyperscaler? | He describes focusing NVIDIA's effort on building its computing platform and supporting companies that can supply cloud services. |
| `dwarkesh-2026-04` | 01:35:06, opening exchange | Could NVIDIA use spare capacity on an older chipmaking node? | He says packaging and system design make such a move costly, while a permanent capacity constraint would change his answer. |

## Acquired

Open the episode's Transcript dialog and find the host's prompt using the
short search cue. The selected blocks identify each speaker in the transcript.

| Exchange ID | Host and search cue | Question in brief | Huang's answer in brief |
| --- | --- | --- | --- |
| `acquired-2023-01` | Ben, `when you have conviction` | Should founders bet the company when they believe in a product? | He pushes back on the framing and recalls emulating the chip and testing the software stack before committing to manufacturing. |
| `acquired-2023-02` | Ben, `was the case with CUDA` | Had NVIDIA tested the idea behind CUDA before committing? | He recalls earlier work on a higher-level interface for GPUs and trials of applications such as medical imaging and image processing. |
| `acquired-2023-03` | Ben, `great 10-year lead` | How can a company keep its lead once competitors arrive? | A product that helps other companies serve the same market can become a platform, with the surrounding ecosystem contributing to its advantage. |
| `acquired-2023-04` | David, `magically 30 years old` | Would he choose to found a company again if he were starting in 2023? | He says he would decline, recalling how much harder founding NVIDIA turned out to be than he had anticipated. |

## Stripe Sessions 2024

Search for the cue in Patrick Collison's prompt on the publisher transcript.

| Exchange ID | Search cue | Question in brief | Huang's answer in brief |
| --- | --- | --- | --- |
| `stripe-sessions-2024-01` | `Anything in your upbringing` | Where did his attitude toward difficult work come from? | He describes his parents' example, chores at boarding school, and early restaurant jobs as experiences that shaped how he approached work. |
| `stripe-sessions-2024-02` | `Is it 60 at every layer` | Should each level copy the CEO's unusually broad span of direct reports? | He says that level of breadth does not extend down the organization because supervision needs differ by role and seniority. |
| `stripe-sessions-2024-03` | `meetings in a typical day` | What meetings does he spend his time in? | He says he leaves routine operations to the people who run them and focuses on problems where his involvement can help. |
| `stripe-sessions-2024-04` | `as successful in AI without CUDA` | Would NVIDIA have had the same AI success without CUDA? | He describes CUDA as essential and explains the economics of accelerating a small number of routines that consume much of a program's runtime. |

## Collection status

These are curated source references and editorial paraphrases for corpus
preparation. The source transcripts contain the original wording. The index has
no training or evaluation split assignment, and review used transcripts without
checking the underlying recordings. Preserve each event ID when making splits
so republished clips cannot appear on both sides. Audio acquisition is outside
this collection.
