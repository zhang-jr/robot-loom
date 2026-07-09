"""Critic tools - embodied progress supervisor.

Scope:
    This package hosts **physical-world progress judgment**. A critic consumes
    a sensor frame (video / image / robot state snapshot) and emits one of
    four states -- ``progress`` / ``completion`` / ``failure`` / ``unchanged``
    -- with a confidence value and supporting evidence. It is a supervisor,
    **not a hard truth**: :class:`ReplanPolicy` combines its verdict with
    sensor-side heuristics before deciding to continue, retry, or replan.

    Typical call rate is 1-3 Hz; calls cross RPC.

Boundary clarification (easy to confuse, MUST stay distinct):

    - **Not the cognitive reflection tool.** The reflection tool under
      ``tools/cognitive/reflection_tool.py`` inspects the LLM's *own* tool
      calls and reasoning and produces self-evaluation. This package looks at
      the *physical world*. The two are orthogonal, may coexist, and must
      never share a name.

    - **Not the verb-level CompletionVerdict.** On-robot verbs exposed by
      ``robot_sdk`` (such as ``reactive_grasp`` / ``visual_servo_to``) report
      their result via ``CompletionVerdict(outcome, evidence, ...)`` returned
      from ``DispatchHandle.await_completion()``. A ``CompletionVerdict`` is
      authoritative on *internal robot state* but limited in viewpoint - it
      knows whether the action finished, not whether the task progressed.
      A critic verdict in this package is the *external supervisor* viewpoint:
      it reads camera frames and judges task-level progress.

      The two form a dual:

          - CompletionVerdict: verb-scoped, authoritative on internal state
          - Critic verdict:    task-scoped, authoritative on external view

      Skills typically consume both. When the two disagree, ``ReplanPolicy``
      reconciles them rather than letting either side stop the task alone.
"""
