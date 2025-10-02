## Finding Semantically Guided Repairs in PDDL Domains Using LLMs

## Nader Karimi Bavandpour, Pascal Bercher

College of Systems &amp; Society, The Australian National University, Australia { Nader.KarimiBavandpour, Pascal.Bercher } @anu.edu.au

## Abstract

Repairing Planning Domain Definition Language (PDDL) models is difficult because solutions must ensure correctness while remaining interpretable to human modellers. Existing hitting set methods identify minimal repair sets from whitelist and blacklist traces, but they cannot prefer semantically meaningful fixes and the true repair may not be minimal. We propose combining large language models (LLMs) with the hitting set framework, using semantic cues in PDDL action and predicate names to guide repairs. This hybrid approach provides contrastive, counterfactual explanations of why traces fail and how domains could behave differently.

## Introduction

Explainability is a central requirement for AI systems that interact with or support humans in decision making. In AI planning, this requirement is naturally addressed by the explicit representation of actions, states, and goals: planners generate solutions by reasoning over structured models of the world. Compared to black-box machine learning techniques, this explicit reasoning process makes planning inherently transparent and interpretable. However, one of the main challenges to deploying planning in practice lies in constructing the planning models themselves (Tantakoun, Zhu, and Muise 2025).

The recent success of Largarge language models (LLMs) has drawn a lot of attention to leverage it in AI planning tasks (Huang, Lipovetzky, and Cohn 2025; Katz et al. 2025; Huang, Cohn, and Lipovetzky 2024; Oswald et al. 2024; Guan et al. 2023). Recent surveys (Tantakoun, Zhu, and Muise 2025) highlight the potential of LLMs to support the construction and refinement of planning models. While verifiable planning modules remain the backbone of reliability, robustness, and explainability, LLMs can act as assistants to reduce the manual burden of defining domain models. We believe that one promising avenue for this is through domain repair , where the goal is to identify a set of modifications to a domain such that a given set of positive traces becomes executable and a set of negative traces becomes inapplicable.

Repairs can themselves be understood as explanations. Following Miller's account of contrastive explanations in the social sciences (Miller 2019), a repair answers the question of why a given trace fails in the current domain, and provides a counterfactual justification of how the domain could have behaved differently. Each repair is thus not only a technical fix, but also a form of interpretable feedback to the human modeller.

Domain repair in planning is an active research area, recently reviewed by (Bercher, Sreedharan, and Vallati 2025). Lin, Grastien, and Bercher (2023) introduced an efficient hitting set algorithm for domain repair with positive (whitelist) traces. It is also possible to use partially lifted test plans if partial information is available (Bavandpour et al. 2025). Lin et al. (2025) extended the hitting-set approach of Lin, Grastien, and Bercher (2023) to handle both positive and negative traces. In this setting, positive traces must remain or become valid plans, while negative (blacklist) traces must be rendered inapplicable. Although effective, optimizationbased approaches, such as the cited works, are limited: they cannot select the semantically most meaningful repair when multiple minimal-cardinality options exist. Moreover, the true repair set may not even be among the minimalcardinality solutions. The VS Code plugin (Lin, Yousefi, and Bercher 2024) which extends the work by Lin, Grastien, and Bercher (2023) is an effort to address this limitation by allowing the human modeller to reject solutions and request the next solution of the hitting-set. However, this is inconvenient and limits automation. Our core idea is to leverage the fact that PDDL domains contain semantically meaningful names for actions and predicates, which LLMs can interpret to guide the repair process. This enables us to move beyond cardinality-minimality toward repairs that are also semantically plausible to human users.

Building on insights from Caglar et al. (2024), who explored LLM applications for model repair but limited their scope to initial state fixes, we propose to combine LLM guidance with the hitting set approach of (Lin et al. 2025). This hybrid aims to exploit semantic knowledge encoded in domain symbols while retaining the optimization guarantees of hitting set solvers. Our experiments evaluate this approach on the same benchmark suite as the baseline paper, allowing for a direct comparison of performance. This paper presents our initial implementation and evaluation of this idea. Extensions and further refinements will be discussed in future work.

## Planning Formalism

Since our focus is on repairing lifted PDDL domains, we introduce the lifted planning formalism. A lifted planning problem is defined as a tuple Π = ( P , A , α, O , s I , G ) , where the domain is D = ( P , A , α ) and the task is T = ( O , s I , G ) .

Objects, Types, and Variables. Let O be the set of objects in the planning task. We consider a set of variables V , each acting as a placeholder for an object. The type of a variable v ∈ V is written as v | t , and the set of objects associated with t is shown by O J t K ⊆ O . We say that t ∈ T is a subtype of t ′ ∈ T iff O J t K ⊆ O J t ′ K .

Predicates and Atoms A predicate symbol P ∈ P and a tuple of k ∈ N 0 typed variables forms an atom p = P ( v 1 | t 1 , . . . , v k | t k ) . We denote by L the set of all atoms in Π .

Variable Substitution. A variable substitution function ϱ : V → O maps each typed variable v | t to an object ϱ ( v | t ) ∈ O J t K of the same type t .

Facts. Given an atom p ∈ L and a substitution function ϱ , a fact is obtained by grounding p , that is, by replacing each parameter ( v 1 , . . . , v k ) with the corresponding objects given by ϱ : f = ϱ ( p ) = P ( ϱ ( v 1 ) , . . . , ϱ ( v k )) . The set of all grounded atoms is denoted by F , and any set of facts constitutes a state . s I and G are called the initial state and the goal description, respectively, each of which is a set of facts.

Action Schemas. Let A denote the set of action schemas. An action schema a = A ( v 1 | t 1 , . . . , v k | t k ) is defined by a unique name A and a tuple of n variables. Each schema is associated with a mapping

<!-- formula-not-decoded -->

whose codomain is (2 L ) 4 , representing a tuple of four sets of compatible atoms, as defined below.

Definition 1 (Compatible Atoms) . For an action schema a , the set of compatible atoms L a contains all atoms whose set of parameters is a subset of the parameters of a . As an example, P ( v 1 | t 1 ) is a compatible atom for the action schema A ( v 1 | t 1 , v 2 | t 2 ) , but Q ( v 3 | t 3 ) is not.

Actions. Given an action schema a and a substitution function ϱ , the corresponding action is obtained by replacing each parameter of a according to ϱ , and is denoted a = a [ ϱ ] . Actions describe transitions in the state space. An action a is applicable in a state s iff prec + ( a ) ⊆ s and prec -( a ) ∩ s = ∅ . Applying an applicable action a in s produces the successor state

<!-- formula-not-decoded -->

which we denote by s → a s ′ .

Throughout this paper, we use boldface (e.g., p , a ) for atoms and action schemas, and regular typeface (e.g., f , a ) for facts and actions.

Solutions. Let γ = ⟨ a 1 , . . . , a k ⟩ be an action sequence. We write s → ∗ γ s ′ to denote that s ′ results from applying γ to s via a state trajectory ⟨ s 0 , . . . , s k ⟩ where s 0 = s , s k = s ′ , and each action is applicable in its preceding state. A solution to a planning problem is an action sequence γ = ⟨ a 1 , . . . , a k ⟩ such that s I → ∗ γ s ′ for some s ′ with G ⊆ s ′ , and each a i is a grounding of some action schema a ∈ A .

## The Repair Problem

We begin by introducing the notation and syntax used to define possible repair operations for a given planning domain. Next, we describe how a set of such repairs can be applied to produce a modified domain. Based on these concepts, we then formalize the domain repair problem in terms of the defined repair operations and a set of positive and negative plans.

In a planning domain D = ( P , A , α ) , an atomic repair is a modification denoted by r J a , p , c, op K . Here, a ∈ A is an action schema, p ∈ P is an atom compatible with a , c ∈ { prec + , prec -, eff + , eff -} indicates whether the change concerns a positive/negative precondition or effect, and op ∈ { + , -} specifies whether the component is added or removed. We write D ⇒ r D ′ to indicate that applying r to D yields D ′ = ( P , A , α ′ ) , where α ′ is resulted by applying r to α .

̸

A repair set δ for a domain is a finite collection of one or more atomic repairs. We say that δ is valid if and only if it contains no two repairs r, r ′ ∈ δ such that one reverses the effect of the other. Specifically, two repairs r = r J a , p , c, op K and r ′ = r ′ J a ′ , p ′ , c ′ , op ′ K are considered to undo each other if a = a ′ , p = p ′ , c = c ′ , and op = op ′ .

Let D be a domain and δ a valid repair set for D . Applying the repairs in δ in any order yields the same modified domain D ′ . We use D ⇒ ∗ δ D ′ to indicate that D ′ is obtained from D by applying the valid repair set δ .

Definition 1 (Domain Repair Problem) . The domain repair problem is defined as a pair R = ( D , T ) , where D denotes a planning domain and T = { T 1 , . . . , T n } for some n ∈ N . Each element T i is a triple (Π i , P i , E i ) . Here, Π i = ( D , T i ) denotes the planning problem; P i is a finite set of positive plans π + k for Π i ; and E i is a finite set of pairs ( π -k , b k ) associated with Π i . Each π -k denotes a negative plan (a sequence of actions considered undesirable) for Π i , and b k is an integer equal or less than the length of π -k which specifies the index of the first action in π -k which must be inapplicable.

Definition 2 (Solution to the Repair Problem) . A solution to R is a valid repair set δ that transforms the original domain D into a modified domain D ′ through the sequence of repair operations D ⇒ ∗ δ D ′ . This repair must satisfy the following: for every index i with 1 ≤ i ≤ n , all positive plans π + ∈ P i are executable in the updated planning task Π ′ i = ( D ′ , T i ) , and each pair ( π -k , b k ) ∈ E i meets the condition that π -k is not a valid plan for Π ′ i , with the action at position b k being the first that cannot be applied.

Figure 1: Different LLM strategies in domain repair to infuse semantic knowledge in search.

<!-- image -->

## Solving the Repair Problem

Our baseline approach (Lin et al. 2025) proposes a sound algorithm based on conditional hitting sets to solve the domain repair problem. They report precision and recall against known ground truth repairs. To obtain ground truth, they perturb IPC domains by randomly adding or removing preconditions and effects, which allows direct computation of these metrics.

## The Baseline Approach

Here, we briefly restate their approach at a level sufficient to motivate our LLM-based extensions. The algorithm executes all provided traces. For each positive trace, one of the unsatisfied preconditions (if any) encountered along the run in each trace is recorded as a flaw . For a negative trace ( π -k , i k ) , a flaw is recorded if the action at index i k in π -k is applicable. If some earlier action at index j &lt; i k is inapplicable, the trace is handled analogously to a positive one: repairs are generated so that the prefix becomes executable and the first inapplicable action occurs exactly at position i k .

For each flaw, the algorithm enumerates a finite set of candidate repairs. The set of candidates for a single flaw is called a conflict ; at least one element of the conflict must be selected to eliminate that flaw. Consider a missing positive precondition p m for action a 3 in the sequence γ = ⟨ a 1 , a 2 , a 3 ⟩ , and suppose a 1 has p m in its negative effects. The repair candidates include: (i) remove the missing precondition from a 3 , that is r 1 J a 3 , p m , prec + , -K for some ϱ so that a 3 = a 3 [ ϱ ] and p m = p m [ ϱ ] ; (ii) add p m to the positive effects of a 2 , that is r 2 J a 2 , p m , eff + , + K whenever a matching ϱ such that a 2 = a 2 [ ϱ ] and p m = p m [ ϱ ] ; exists; (iii) remove p m from the negative effects of a 1 , that is r 3 J a 1 , p m , eff -, -K for some ϱ so that a 1 = a 1 [ ϱ ] and p m = p m [ ϱ ] ;. These give a conflict θ = { r 1 , r 2 , r 3 } .

For negative traces, conflicts are constructed so that the target action becomes inapplicable at its specified position. For example, one can add a new required precondition q that is false in the corresponding state, or remove earlier effects that would otherwise establish an existing precondition.

Different conflicts may interact. Some repairs may contradict one another, or jointly re-introduce earlier flaws. The baseline therefore augments the plain collection of conflicts to a more complex object Θ that encodes applicability conditions and mutual exclusions, producing a conditional hitting set instance. Note that the details of forming Θ is out of scope here. A solver then returns a minimal-cardinality hitting set, called a diagnosis . The algorithm iterates: it (1) extracts flaws from the current domain using the given positive and negative traces, (2) constructs Θ , (3) solves for a diagnosis, and (4) applies the corresponding repairs to obtain a modified domain. The process repeats until all positive traces succeed and all negative traces fail at their specified positions.

The baseline does not exploit semantic cues encoded by the modeller in predicate, action, or domain names within the PDDL file. Its hitting-set solver optimizes only the size of the repair set; when multiple diagnoses share the same cardinality, it returns an arbitrary one. Moreover, the ground truth repair need not be cardinality-minimal, so it can be missed under this objective.

## LLM-Guided Search

Wepropose five strategies for using LLMs to steer the search toward human-plausible repairs, as Figure 1 shows. Because a planning domain can admits multiple semantically consistent fixes, precision and recall against the ground truths should be viewed as proxies for alignment with modeller intent rather than definitive correctness. Note that ideas 1 and 2 are implemented, but only Idea 1 is evaluated here. Ideas 3-5 are deferred to follow-up work.

Idea 1. This simple approach treats the LLM as a knowledge-engineering assistant. We specify the allowed repair operators, show the full domain to preserve global context, and ask the model to summarize the semantics it infers from action and predicate names. It then proposes a small set of repairs with brief rationales. This variant is implemented, and its results are reported in the next section.

Idea 2. Extract conflicts from action traces as in the base-

Table 1: Experimental results comparing idea #1 with the baseline algorithm. Summary statistics over 5 runs per domain. Words and Lines count PDDL size; Tasks is the number of planning problems; POS-Sum / NEG-Sum are positive/negative plan counts; POS-Len / NEG-Len are their average lengths; Flaws and Repairs are detected issues and fixes; Prec / Rec denote precision and recall.

|               |       |       |       |         |         |         |         |       | Idea #1   | Idea #1   | Idea #1   | Baseline   | Baseline   | Baseline   |
|---------------|-------|-------|-------|---------|---------|---------|---------|-------|-----------|-----------|-----------|------------|------------|------------|
| Domain        | Words | Lines | Tasks | POS-Sum | NEG-Sum | POS-Len | NEG-Len | Flaws | Repairs   | Prec      | Rec       | Repairs    | Prec       | Rec        |
| FLOORTILE     | 348   | 109   | 20    | 1       | 5       | 27.00   | 1.00    | 4     | 3.20      | 1.00      | 0.75      | 2.00       | 0.70       | 0.35       |
| FREECELL      | 597   | 215   | 80    | 62      | 60      | 43.15   | 38.00   | 4     | 4.00      | 0.56      | 0.50      | 3.00       | 0.73       | 0.55       |
| GED           | 784   | 310   | 20    | 20      | 9       | 14.30   | 8.67    | 10    | 4.00      | 0.13      | 0.06      | 4.00       | 0.90       | 0.36       |
| HIKING        | 442   | 140   | 20    | 7       | 2       | 20.43   | 13.50   | 4     | 2.60      | 0.73      | 0.45      | 2.00       | 0.40       | 0.20       |
| LOGISTICS00   | 282   | 99    | 28    | 28      | 1       | 47.54   | 1.00    | 4     | 3.60      | 0.90      | 0.75      | 2.00       | 0.50       | 0.25       |
| LOGISTICS98   | 276   | 97    | 35    | 27      | 6       | 67.96   | 1.33    | 4     | 3.40      | 0.85      | 0.70      | 3.00       | 0.60       | 0.45       |
| MPRIME        | 307   | 97    | 35    | 30      | 29      | 11.10   | 4.66    | 2     | 3.60      | 0.15      | 0.30      | 2.00       | 1.00       | 1.00       |
| SCANALYZER    | 323   | 94    | 20    | 14      | 1       | 42.57   | 4.00    | 2     | 1.20      | 0.90      | 0.50      | 2.00       | 0.70       | 0.70       |
| SLITHERLINK   | 410   | 133   | 20    | 1       | 2       | 19.00   | 14.00   | 2     | 2.40      | 0.53      | 0.60      | 2.00       | 0.50       | 0.50       |
| SOKOBAN       | 239   | 77    | 20    | 2       | 3       | 40.50   | 17.67   | 2     | 2.80      | 0.43      | 0.60      | 2.00       | 1.00       | 1.00       |
| TETRIS        | 477   | 144   | 17    | 5       | 5       | 24.60   | 20.80   | 4     | 3.20      | 0.29      | 0.20      | 2.00       | 0.80       | 0.40       |
| THOUGHTFUL    | 1283  | 469   | 20    | 15      | 1       | 135.27  | 2.00    | 10    | 4.80      | 0.32      | 0.14      | 3.00       | 0.53       | 0.16       |
| TIDYBOT       | 2157  | 656   | 20    | 4       | 5       | 29.25   | 19.20   | 12    | 3.60      | 0.60      | 0.17      | 2.00       | 1.00       | 0.17       |
| WOODWORKING08 | 972   | 300   | 30    | 30      | 5       | 20.77   | 6.00    | 6     | 3.60      | 0.73      | 0.40      | 7.00       | 0.17       | 0.20       |
| WOODWORKING11 | 976   | 302   | 20    | 7       | 1       | 62.29   | 9.00    | 6     | 2.80      | 0.53      | 0.23      | 3.00       | 0.53       | 0.27       |

line, take their union, and iteratively ask the LLM to choose the most plausible repair, apply it to the domain, and repeat. Note that by taking union of the conflicts we are creating a long flat set of repairs, each of which fix at least one of the flaws found in the action traces. While we are not ready to report on its performance, preliminary tests suggest that the flat candidate pool can become large, which degrades LLM selection quality and makes this approach less promising.

Idea 3. Combine LLM proposals with minimality via the hitting-set solver. After exposing the full domain for context, we query the LLM for a list of plausible repairs at the action level. The hitting-set solver then selects a cardinalityminimal subset, ensuring that the final repairs are optimized toward consistency across conflicts. This separation allows the LLM to focus on local, per-action suggestions, avoiding confusion from reasoning over global repair sets, while the optimization stage consolidates them. This could improve the results compared to Idea 1, with the added benefit of shorter and more efficient prompts.

Idea 4. Asymmetric decomposition to Idea 3 that operates at the level of conflicts rather than actions. For each flaw, the LLM filters or ranks its candidate repairs, and we keep conflict sets separate instead of forming a single union. This yields more focused prompts and allows us to test whether action-centric (Idea 3) or flaw-centric filtering is more natural for LLMs.

Idea 5. Use the LLM as a post-hoc judge. The baseline algorithm proposes a minimal repair; the LLM then accepts or vetoes it on semantic grounds and requests alternatives if needed. This mirrors the human-in-the-loop design of the VS Code plugin (Lin, Yousefi, and Bercher 2024) for the predecessor algorithm (Lin, Grastien, and Bercher 2023), with the LLM replacing the human reviewer.

## Experiments

We use the same problem set as the baseline paper (Lin et al. 2025) to ensure a direct comparison, and we report the same metrics of precision and recall. Runtime results are omitted for now, as they are negligible (comparable to the latency of a single OpenAI API call) but will be included in future work. The baseline numbers in our tables are taken directly from the published paper; we did not rerun their implementation locally. Our experiments employ the model gpt-4o-2024-08-06 , where the suffix denotes the training cutoff date.

Despite its simplicity, Idea 1 improves both precision and recall in roughly half of the benchmark domains. We also report word and line counts of the PDDL files as a proxy for domain complexity, although no clear correlation with LLM performance emerges. Crucially, Idea 1 does not exploit information from action traces, so we expect that incorporating our other proposed methods will further improve results across all domains. An additional insight concerns the number of repairs returned by the LLM: it is lower than the baseline in some domains and higher in others. When fewer repairs are produced, combining with the baseline is necessary, as any sound repair set necessarily has a cardinality greater than or equal to the minimum-cardinality set. When more repairs are produced, recall does not consistently improve over the baseline, which indicates unexploited information in the action traces. Trace availability and length vary by domain: some domains include 62 traces with an average length of 43 steps, whereas a different domain has a single trace of length 135. Given this sizes, we do not expect the LLM alone to exploit trace information reliably, and we believe integrating the LLM with hitting-set methods that consume traces is therefore essential for consistent gains.

## Conclusion &amp; Future Work

We introduced the use of large language models (LLMs) to exploit semantic cues in PDDL domains, guiding repairs toward solutions that are preferable for human modellers. Our approach leverages action and predicate names to improve semantic alignment. Among five proposed strategies, a simple single-shot prompt outperformed the baseline on half the benchmark domains. A limitation is that benchmarks may overlap with LLM training data; future work will test on unpublished domains for a more reliable evaluation.

## Acknowledgements

Pascal Bercher is the recipient of an Australian Research Council (ARC) Discovery Early Career Researcher Award (DECRA), project number DE240101245, funded by the Australian Government.

## References

Bavandpour, N. K.; Lauer, P.; Lin, S.; and Bercher, P. 2025. Repairing Planning Domains Based on Lifted Test Plans. In Proc. of the 28th ECAI .

Bercher, P.; Sreedharan, S.; and Vallati, M. 2025. A Survey on Model Repair in AI Planning. In Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025) .

Caglar, T.; Belhaj, S.; Chakraborti, T.; Katz, M.; and Sreedharan, S. 2024. Can LLMs Fix Issues with Reasoning Models? Towards More Likely Models for AI Planning. In Proc. of the 38th AAAI , 20061-20069.

Guan, L.; Valmeekam, K.; Sreedharan, S.; and Kambhampati, S. 2023. Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning. In Proc. of the 37th NeurIPS .

Huang, S.; Cohn, T.; and Lipovetzky, N. 2024. Chasing Progress, Not Perfection: Revisiting Strategies for End-toEnd LLM Plan Generation. CoRR .

Huang, S.; Lipovetzky, N.; and Cohn, T. 2025. Planning in the Dark: LLM-Symbolic Planning Pipeline Without Experts. In Proc. of the 39th AAAI , 26542-26550.

- Katz, M.; Kokel, H.; Muise, C.; Sohrabi, S.; and Sreedharan, S. 2025. Make Planning Research Rigorous Again! CoRR , abs/2505.21674.
- Lin, S.; Grastien, A.; and Bercher, P. 2023. Towards Automated Modeling Assistance: An Efficient Approach for Repairing Flawed Planning Domains. In Proc. of the 37th AAAI , 12022-12031.
- Lin, S.; Grastien, A.; Shome, R.; and Bercher, P. 2025. Told You That Will Not Work: Optimal Corrections to Planning Domains Using Counter-Example Plans. In Proc. of the 39th AAAI , 26596-26604.
- Lin, S.; Yousefi, M.; and Bercher, P. 2024. A Visual Studio Code Extension for Automatically Repairing Planning Domains. In Demo at the 34th ICAPS .
- Miller, T. 2019. Explanation in artificial intelligence: Insights from the social sciences. AIJ , 267: 1-38.

Oswald, J. T.; Srinivas, K.; Kokel, H.; Lee, J.; Katz, M.; and Sohrabi, S. 2024. Large Language Models as Planning Domain Generators. In Proc. of the 34th ICAPS , 423-431.

Tantakoun, M.; Zhu, X.; and Muise, C. 2025. LLMs as Planning Modelers: A Survey for Leveraging Large Language Models to Construct Automated Planning Models. CoRR .