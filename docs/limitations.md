# Limitations

This repository is a research baseline and simulator. It does not include perception, localisation, airport operations integration, safety certification, redundancy management, hardware actuation, braking systems, fail-operational planning, or formal verification of a deployed controller.

The simulator uses a standard planar kinematic chain. Real articulated vehicles have tyre slip, backlash, hitch compliance, actuator delays, steering dead zones, state-estimation error, wheel-speed constraints, and surface-dependent friction. Those effects should be identified from instrumented vehicle data before any physical experiment.

The learned policy is trained from a sampling-based expert in simulation. It must therefore be treated as an approximation to a simulated controller, not as proof of safe real-world capability. For physical systems, place a certified low-level controller, barrier constraints, emergency stop, speed governors, geofencing, and human supervision outside any learned component.
