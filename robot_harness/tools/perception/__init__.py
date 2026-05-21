"""Perception tools - slow-loop grounding perception only.

Scope:
    This package hosts **slow-loop semantic perception** only. Use it for
    one-shot scene snapshots, language-to-image grounding, and producing hints
    that the harness then hands to an on-robot verb. Typical call rate is the
    Brain rate (0.5-7 Hz); calls cross RPC and can be served by a single
    perception server shared across robots.

    Typical usage:
        - ``perception.detect_objects(prompts=['mug', 'plate'])`` -- scene check
        - ``perception.ground_phrase('the red mug')``             -- phrase -> bbox
        - ``perception.estimate_depth(image)``                    -- depth probe

Out of scope (lives on the robot, not in this package):
    All **mid/tight-loop reactive perception** -- visual servo ticks, depth
    servo, online grasp micro-adjustment, contact-force fusion, and anything
    else at 5-30 Hz or above that forms a perception-action loop. These belong
    inside on-robot verbs such as ``reactive_grasp`` / ``visual_servo_to``.
    They run inside the per-robot ``agent_server``, are invisible to the
    harness, and surface only as a verb-level ``CompletionVerdict`` returned
    by ``DispatchHandle.await_completion()``.

    Do not call any tool in this package on a 30 Hz tick - that is the exact
    anti-pattern this boundary exists to prevent.
"""
