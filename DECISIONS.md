# Decision log (Madhvansh's rulings)

Kept so the README's account of AI use is accurate and so every call can be defended in the interview. Times are IST. The AI (Claude Code sessions) put options to me and wrote the code; these are the calls I made.

## 21 Nov 2026

D1, about 13:30. Central design.
Options put to him: (a) recovered-formula graph with join, verify, derive; (b) lighter version with joins and typed plans only; (c) a lazy graph that learns from questions through a gate, as in Genloop's paper.
His ruling: take the most impressive design and do not drop anything for being hard. Combine (a) and (c): the recovered-formula graph as the verified base, with a living layer on top that learns semantics from questions through an admission gate. It must be robust and general, not tuned to a few companies.

D2, about 13:30. Corpus.
His ruling: about 40 latest 10-Ks across sectors (banks included), plus the prior-year 10-K for about 10 of them so restatements show up as conflicts between filings, plus a few 10-Qs so a quarter that is printed nowhere can be derived.

D3, about 13:30. Fine-tuning.
The prior-art survey and the red team both advised skipping it. His ruling: do it regardless. A small local model is fine-tuned to turn questions into query plans. Kaggle or Colab may be used.

D4, about 13:30. Name.
His ruling: TieOut.

D5, about 13:30. AI use.
Genloop's brief says "Use AI tools as much as you like, just tell us in the README how you used them". His reading: the take-home uses AI openly and the README says how. The interview is his alone.

D6, about 15:25. Resources.
He created the free Gemini API key and wrote the SEC contact line into `.env` himself (`.env` is ignored by git).

D7, about 18:15. Publication.
Asked whether to create the public repo `Madhvansh/tieout` and push the local commits. His ruling: "Yes, push it." Tag `frozen-before-benchmark` pushed with it.

D8, about 19:45. Manual checks.
Asked to check the 30 gold answers and review the learned bindings himself (about 75 minutes). His ruling: no; do not build things that make work for him. The checks are to be done by an automated pass, and the README must say that and nothing more.

D9, about 19:50. Stop and hand off.
His ruling: stop everything; he is doing a revamp in another session.

D16, about 21:40 (recorded later). The headline and the submission.
The headline number is the round-2 held-out set, run once after the tag `frozen-v2`, with the history of round 1 beside it. On Monday the AI fills the submission form in his browser and stops; he clicks Submit himself.

## 22 Nov 2026

D10, about 12:45. Third red team and a full revamp.
He stopped the two overnight sessions and ordered a takeover with fresh eyes: red-team everything the earlier sessions built and claimed, write down what failed for them, and revamp rather than patch. He repeated that bias toward earlier conclusions is not acceptable.

D11, about 12:45. Hidden-question performance is the target.
He asked for the best accuracy on questions nobody has seen, not on the development set. This is the ruling behind the fresh blind rounds that measured the overfitting.

D12, about 13:00 and 13:40. Resources.
He supplied more Gemini and Groq keys and Kaggle accounts, and ruled that the session may run anything it needs on the laptop. Extra keys are for development only, so the reported run fits one free key.

D13, about 13:30. Research before training, and research on the people.
He repeated the ruling of the night before: no blind training; deep research first, on Genloop and its people and on the topic, then several research-backed experiments at once.

D14, about 13:45. The lazy graph stays.
He asked where the learned (lazy) graph of his first ruling D1 was in the revamp. It stays as the layer that learns links from questions; the revamp adds what it lacked: a second witness before a link is trusted and a way to withdraw a link.

D15, about 13:50. Do the hard thing.
He ruled again that difficulty and time are not reasons to drop an idea: step back, learn from the failure, try another way.

D17, about 22:55. Free headline, paid models only as a test.
The brief says no paid services are needed. His ruling: the headline runs on the free configuration (one free Gemini key and the fine-tuned 4B planner on the laptop CPU). Paid models are used only for development throughput and for one labelled side experiment that tests whether the model is the bottleneck, never mixed into a headline number.

## 23 Nov 2026

D18, about 02:45. Answers come from the graph.
The last night's revamp had added a reader of retrieved passages that served some answers itself. His ruling, quoting the brief ("Answers must come from the graph, not from plain vector search over text chunks"): as a witness that only corroborates or disputes an answer of the graph it is fine; as the source of a served answer it is not, unless the figure it finds is resolved to a node of the graph and the answer is read from that node. The reader was rebuilt that way: a witness and a pointer, never a source.
