import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

# --------------------------
# Parameters
# --------------------------
cube_size_x = 250
cube_size_y = 250
cube_size_z = 150  
dt = 0.1
total_time = 80.0  # seconds

# --------------------------
# Terrain (randomized hills)
# --------------------------
num_hills = 5
hill_params = []
for _ in range(num_hills):
    cx = np.random.uniform(cube_size_x*0.1, cube_size_x*0.9)
    cy = np.random.uniform(cube_size_y*0.1, cube_size_y*0.9)
    
    height = np.random.uniform(30, 60)
    radius = np.random.uniform(25, 50)
    hill_params.append((cx, cy, height, radius))

x_grid = np.linspace(0, cube_size_x, 150)
y_grid = np.linspace(0, cube_size_y, 150)
X, Y = np.meshgrid(x_grid, y_grid)
Z_terrain = np.zeros_like(X)

for cx, cy, height, radius in hill_params:
    Z_hill = height * np.maximum(0, 1 - ((X - cx)**2 + (Y - cy)**2) / radius**2)
    Z_terrain = np.maximum(Z_terrain, Z_hill)

# Pre-calculate terrain gradient for the entire map
grad_y, grad_x = np.gradient(Z_terrain, y_grid[1]-y_grid[0], x_grid[1]-x_grid[0])

# --------------------------
# Aircraft Initial State (centered spawn point)
# --------------------------
pos = np.array([
    cube_size_x / 2,      # Center of X range
    cube_size_y / 2,      # Center of Y range
    cube_size_z / 2       # Center of Z range 
])

# Ensure aircraft starts at least 2m above the terrain
terrain_idx_x = np.abs(x_grid - pos[0]).argmin()
terrain_idx_y = np.abs(y_grid - pos[1]).argmin()
terrain_z_start = Z_terrain[terrain_idx_y, terrain_idx_x]
pos[2] = max(pos[2], terrain_z_start + 5)

# Create a random initial direction
init_dir = np.random.randn(3)
# Ensure the z-component is non-negative (no downward initial velocity)
init_dir[2] = abs(init_dir[2]) 
init_dir = normalize(init_dir)

init_speed = np.random.uniform(5, 10)
velocity = init_dir * init_speed

# Physics
k_lift = 0.015 # Lift constant
k_drag = 0.005 # Drag constant
g_vector = np.array([0, 0, -9.8])
mass = 0.75 # kg
thrust = 10.0 # Newtons

# Speed control parameters
V_setpoint = 30.0  # m/s, desired speed ceiling
Kp_thrust = 1.0    # proportional gain
Kp_soa_brake = 1.0 # Proportional gain for SOA-based braking

# --- Roll Control Parameters ---
max_roll_target_deg = 80.0    # Clamp for commanded roll angle
Kp_roll_correction = 2.0      # Proportional gain for roll correction (restores level flight)
Kd_roll = 1.0                 # Derivative gain to dampen roll rate

# --- Pitch Control Parameters ---
min_pitch_deg = -60.0         # The pitch angle below which correction starts.
max_pitch_deg = 60.0          # The pitch angle above which correction starts.
Kp_pitch = 2.0                # Proportional gain for pitch correction.
Kd_pitch = 2.0                # Derivative gain to dampen pitch rate changes.
pitch_rate_scaling_max_rad_s = np.radians(120) # Pitch rate at which AoA effect is max (120 deg/s)

# --- Sphere of Awareness (SOA) Parameters ---
soa_radius = 35.0             # Radius of awareness sphere in meters
Kp_soa_pitch = 2.0            # Gain for vertical SOA avoidance (pitch response)
Kp_soa_roll = 3.0             # Gain for horizontal SOA avoidance (roll response)

positions = [pos.copy()]
velocities = [velocity.copy()]
speeds = [np.linalg.norm(velocity)]
ribbon_normals = []
prev_signed_roll_rad = 0.0 # For derivative control
phi_cmd_rad = 0.0 # Stateful commanded roll angle
roll_angles_rad = [] # Store commanded roll angles for statistics

# --- Pitch State Variables ---
prev_pitch_rad = 0.0 # For derivative control

# --- Stable Reference Frame State ---
# Initialize a stable local reference frame for the aircraft
v_norm_init = normalize(velocity)
# Handle the edge case of a perfectly vertical initial velocity
if np.abs(np.dot(v_norm_init, np.array([0, 0, 1]))) > 0.999:
    # If vertical, use world Y as the initial 'right' reference
    local_right = normalize(np.cross(np.array([0, 1, 0]), v_norm_init))
else:
    # Otherwise, use world Z to define the initial 'right' vector
    local_right = normalize(np.cross(v_norm_init, np.array([0, 0, 1])))
local_up = normalize(np.cross(local_right, v_norm_init))


# Global Z-axis for roll angle calculation
z_world = np.array([0, 0, 1])

# Storage for lift vectors at every 10th timestep
lift_vector_positions = []
lift_vectors = []
timestep_counter = 0

# Storage for SOA visualization
soa_positions = []
soa_activations = []  # Store when SOA is active

# Storage for avoidance vector visualization
avoidance_vector_positions = []
avoidance_vectors = []

# Storage for final pre-impact vectors for analysis
final_lift_vec = np.zeros(3)
final_avoidance_vec = np.zeros(3)
lift_vectors_history = [] # Store lift vector at each timestep for analysis

t = 0.0 # Initialize time variable
for t in np.arange(dt, total_time, dt):
    timestep_counter += 1
    
    speed = np.linalg.norm(velocity)
    if speed == 0:
        break
    
    # --- Calculate Pitch Rate for AoA Surrogate ---
    # This must be done before drag/lift calculations
    v_norm_early = normalize(velocity)
    current_pitch_rad = np.arcsin(np.clip(v_norm_early[2], -1.0, 1.0))
    pitch_rate = (current_pitch_rad - prev_pitch_rad) / dt
    # prev_pitch_rad is updated later in the main pitch controller
    
    # --- AoA Surrogate Scaling ---
    # Scale lift/drag coefficients based on pitch rate magnitude.
    # This simulates increased AoA during pitching maneuvers.
    normalized_pitch_rate = np.clip(np.abs(pitch_rate) / pitch_rate_scaling_max_rad_s, 0.0, 1.0)
    aoa_surrogate_factor = 0.5 * normalized_pitch_rate # 0.0 to 0.5 scaling
    
    k_lift_dynamic = k_lift * (1.0 + aoa_surrogate_factor)
    k_drag_dynamic = k_drag * (1.0 + aoa_surrogate_factor)
    
    # Speed control: reduce thrust if above setpoint
    thrust_cmd = thrust
    if speed > V_setpoint:
        thrust_cmd = thrust - Kp_thrust * (speed - V_setpoint)
    
    drag_force = k_drag_dynamic * speed**2
    drag_accel = drag_force / mass
    drag_vec = -normalize(velocity) * drag_accel

    # Get current terrain height at aircraft position
    terrain_idx_x = np.abs(x_grid - pos[0]).argmin()
    terrain_idx_y = np.abs(y_grid - pos[1]).argmin()
    terrain_z = Z_terrain[terrain_idx_y, terrain_idx_x]

    # --- Calculate terrain normal from pre-computed gradient ---
    dz_dx = grad_x[terrain_idx_y, terrain_idx_x]
    dz_dy = grad_y[terrain_idx_y, terrain_idx_x]
    terrain_normal = normalize(np.array([-dz_dx, -dz_dy, 1.0]))

    # --- Unified Sphere of Awareness (SOA) with Avoidance Vector ---
    max_penetration = 0.0
    avoidance_vec = np.array([0.0, 0.0, 0.0])

    # Define all potential threats: (closest_surface_point, surface_normal_vector)
    threats = [
        (np.array([pos[0], pos[1], terrain_z]),   terrain_normal),               # Terrain
        (np.array([pos[0], pos[1], cube_size_z]), np.array([0.0, 0.0, -1.0])),  # Ceiling
        (np.array([pos[0], 0, pos[2]]),           np.array([0.0, 1.0, 0.0])),   # Y=0 Wall
        (np.array([pos[0], cube_size_y, pos[2]]), np.array([0.0, -1.0, 0.0])),  # Y=max Wall
        (np.array([0, pos[1], pos[2]]),           np.array([1.0, 0.0, 0.0])),   # X=0 Wall
        (np.array([cube_size_x, pos[1], pos[2]]), np.array([-1.0, 0.0, 0.0])),  # X=max Wall
    ]
    
    for surface_point, normal_vec in threats:
        dist_vec = pos - surface_point
        distance_to_surface = np.dot(dist_vec, normal_vec)
        
        if distance_to_surface < soa_radius:
            penetration = soa_radius - distance_to_surface
            if penetration > max_penetration:
                max_penetration = penetration
                avoidance_vec = normal_vec

    soa_active = (max_penetration > 0)
    
    # Store SOA data for visualization
    soa_activations.append(soa_active)
    if soa_active:
        soa_positions.append(pos.copy())
    
    if soa_active:
        avoidance_vector_positions.append(pos.copy())
        avoidance_vectors.append(avoidance_vec * 7.0)
    
        # --- SOA-based Braking ---
        # Check if the avoidance vector is retrograde to the velocity (i.e., we are flying towards the threat)
        cos_theta_avoid = np.dot(normalize(velocity), avoidance_vec)
        if cos_theta_avoid < 0: # Heading towards the threat
            normalized_penetration = max_penetration / soa_radius
            # Apply braking force proportional to penetration and how directly we're heading towards the threat
            braking_force = Kp_soa_brake * normalized_penetration * abs(cos_theta_avoid) * thrust
            thrust_cmd -= braking_force

    # Clamp thrust command so it cannot be negative (no reverse thrust).
    thrust_cmd = max(0.0, thrust_cmd)
    
    # Calculate final thrust acceleration *after* all modifications (speed control, SOA braking).
    thrust_accel = (thrust_cmd / mass) * normalize(velocity)
    
    # --- Roll Controller with SOA Integration ---
    v_norm = normalize(velocity)
    
    # --- Continuous Reference Frame Calculation (Gram-Schmidt) ---
    # This method avoids gimbal lock by incrementally updating the local frame
    # based on its previous state, ensuring smoothness.
    
    # 1. Project the previous 'up' vector onto the plane perpendicular to the new velocity.
    # This finds the 'up' direction that is closest to the previous one.
    local_up = local_up - np.dot(local_up, v_norm) * v_norm
    local_up = normalize(local_up)
    
    # 2. The 'right' vector is orthogonal to both velocity and the new 'up' vector.
    local_right = np.cross(v_norm, local_up)
    
    # The aircraft's "level" vectors are now defined by this stable local frame.
    level_up_vec = local_up
    level_right_vec = local_right
    # --- End Continuous Reference Frame Calculation ---

    # Calculate Pitch State (theta) - using the already computed value
    pitch_angle_rad = current_pitch_rad

    # Roll PD Control with SOA integration
    target_roll_angle_rad = 0.0
    if soa_active:
        # Project avoidance vector onto the aircraft's local horizontal plane
        # to determine the desired roll direction.
        roll_component = np.dot(avoidance_vec, level_right_vec)
        normalized_penetration = max_penetration / soa_radius
        
        # Command a roll proportional to the horizontal component of the avoidance vector
        soa_target_roll_rad = Kp_soa_roll * normalized_penetration * roll_component
        target_roll_angle_rad = np.clip(soa_target_roll_rad, 
                                        np.radians(-max_roll_target_deg), 
                                        np.radians(max_roll_target_deg))

    # P-Term (Proportional)
    error_p = target_roll_angle_rad - phi_cmd_rad
    
    # D-Term (Derivative)
    roll_rate = (phi_cmd_rad - prev_signed_roll_rad) / dt
    prev_signed_roll_rad = phi_cmd_rad

    # PD controller output
    phi_cmd_rad += (Kp_roll_correction * error_p - Kd_roll * roll_rate) * dt
    phi_cmd_rad = np.clip(phi_cmd_rad, np.radians(-max_roll_target_deg), np.radians(max_roll_target_deg))
    roll_angles_rad.append(phi_cmd_rad) # Store commanded roll angle

    # Rebuild the 'actual_binormal' from the commanded roll angle
    actual_binormal = np.cos(phi_cmd_rad) * level_up_vec + np.sin(phi_cmd_rad) * level_right_vec
    actual_binormal = normalize(actual_binormal)
    binormal = actual_binormal 
    
    # Wing span vector for ribbon visualization
    wing_span_vector = np.cos(phi_cmd_rad) * level_right_vec - np.sin(phi_cmd_rad) * level_up_vec
    wing_span_vector = normalize(wing_span_vector)
    ribbon_normals.append(wing_span_vector)
    
    # --- Pitch PD Controller with SOA Integration ---
    lift_force = k_lift_dynamic * speed**2
    base_lift_vec = normalize(binormal) * (lift_force / mass)

    pitch_error = 0.0
    if pitch_angle_rad < np.radians(min_pitch_deg):
        pitch_error = np.radians(min_pitch_deg) - pitch_angle_rad
    elif pitch_angle_rad > np.radians(max_pitch_deg):
        pitch_error = np.radians(max_pitch_deg) - pitch_angle_rad
        
    # The pitch rate is already calculated at the top of the loop for AoA scaling.
    # We just need to update the state for the next timestep's derivative calculation.
    prev_pitch_rad = pitch_angle_rad

    pitch_limits_correction = Kp_pitch * pitch_error - Kd_pitch * pitch_rate
    
    soa_pitch_correction = 0.0
    if soa_active:
        pitch_component = np.dot(avoidance_vec, binormal)
        normalized_penetration = max_penetration / soa_radius
        soa_pitch_correction = Kp_soa_pitch * normalized_penetration * pitch_component

    total_pitch_correction = pitch_limits_correction + soa_pitch_correction
    
    pitch_correction_vec = binormal * total_pitch_correction
    
    lift_vec = base_lift_vec + pitch_correction_vec

    # --- ORTHOGONALITY CORRECTION ---
    # Project lift vector onto plane perpendicular to velocity to maintain physical realism
    velocity_norm = normalize(velocity)
    lift_vec = lift_vec - np.dot(lift_vec, velocity_norm) * velocity_norm
    # --- END ORTHOGONALITY CORRECTION ---

    # Store the lift vector for post-crash analysis
    lift_vectors_history.append(lift_vec.copy())

    # Store the final vectors for post-mortem analysis
    final_lift_vec = lift_vec.copy()
    if soa_active:
        final_avoidance_vec = avoidance_vec.copy()
    else:
        final_avoidance_vec = np.zeros(3)


    if timestep_counter % 2 == 0:
        lift_vector_positions.append(pos.copy())
        lift_vectors.append(normalize(lift_vec) * 5)

    net_accel = thrust_accel + drag_vec + lift_vec + g_vector
    velocity = velocity + net_accel * dt
    pos = pos + velocity * dt

    # Boundary collision detection
    if pos[0] <= 0 or pos[0] >= cube_size_x or pos[1] <= 0 or pos[1] >= cube_size_y:
        positions.append(pos.copy())
        velocities.append(np.zeros(3))
        speeds.append(0)
        break
    
    if pos[2] <= terrain_z:
        pos[2] = terrain_z
        positions.append(pos.copy())
        velocities.append(np.zeros(3))
        speeds.append(0)
        break
    if pos[2] >= cube_size_z:
        pos[2] = cube_size_z
        positions.append(pos.copy())
        velocities.append(np.zeros(3))
        speeds.append(0)
        break

    positions.append(pos.copy())
    velocities.append(velocity.copy())
    speeds.append(np.linalg.norm(velocity))

positions = np.array(positions)
speeds = np.array(speeds)
ribbon_normals = np.array(ribbon_normals)
ribbon_width = 5.0

# Convert data to arrays
lift_vector_positions = np.array(lift_vector_positions)
lift_vectors = np.array(lift_vectors)
soa_positions = np.array(soa_positions)
soa_activations = np.array(soa_activations)
avoidance_vector_positions = np.array(avoidance_vector_positions)
avoidance_vectors = np.array(avoidance_vectors)
velocities = np.array(velocities) # Ensure velocities is a numpy array for later use

# --------------------------
# Path coloring by speed
# --------------------------
points = positions
segments = np.stack([points[:-1], points[1:]], axis=1)
if len(speeds) > 1:
    norm = Normalize(vmin=speeds.min(), vmax=speeds.max())
else:
    norm = Normalize(vmin=0, vmax=V_setpoint)
cmap = plt.get_cmap('coolwarm')
colors = cmap(norm(speeds[:-1]))
lc = Line3DCollection(segments, colors=colors, linewidth=2)

# Ribbon
if len(ribbon_normals) > 1 and len(positions) > 1:
    if ribbon_normals.shape[0] < positions[:-1].shape[0]:
        ribbon_normals = np.vstack([ribbon_normals, ribbon_normals[-1]])

    edge1 = positions[:-1] - ribbon_normals * ribbon_width / 2
    edge2 = positions[:-1] + ribbon_normals * ribbon_width / 2
    verts = [[edge1[i], edge2[i], edge2[i+1], edge1[i+1]] for i in range(len(edge1) - 1)]
    ribbon = Poly3DCollection(verts, facecolors=colors, alpha=0.9)
else:
    ribbon = Poly3DCollection([], alpha=0.9)

# --------------------------
# Plotting
# --------------------------
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Terrain
ax.plot_surface(X, Y, Z_terrain, alpha=0.9, color='tan', linewidth=0, antialiased=True)

# Flight path and ribbon
ax.add_collection3d(lc)
ax.add_collection3d(ribbon)

# Lift vectors
if len(lift_vector_positions) > 0:
    ax.quiver(lift_vector_positions[:, 0], lift_vector_positions[:, 1], lift_vector_positions[:, 2],
              lift_vectors[:, 0], lift_vectors[:, 1], lift_vectors[:, 2], 
              color='magenta', alpha=0.8, arrow_length_ratio=0.1, linewidth=2, label='Lift Vectors')

# Avoidance vectors
if len(avoidance_vector_positions) > 0:
    ax.quiver(avoidance_vector_positions[:, 0], avoidance_vector_positions[:, 1], avoidance_vector_positions[:, 2],
              avoidance_vectors[:, 0], avoidance_vectors[:, 1], avoidance_vectors[:, 2],
              color='cyan', alpha=0.9, arrow_length_ratio=0.15, linewidth=2.5, label='Avoidance Vector')

# SOA visualization - show spheres where SOA was active
if len(soa_positions) > 0:
    ax.scatter(soa_positions[:, 0], soa_positions[:, 1], soa_positions[:, 2],
              color='red', alpha=0.7, s=50, label=f'SOA Active (r={soa_radius}m)')

# Start and end points
ax.scatter(*positions[0], color='g', label='Start', s=70)
ax.scatter(*positions[-1], color='r', label='End', s=70)

# Dummy line for ribbon legend
ax.plot([], [], [], color='gray', linestyle='-', alpha=0.6, label='Ribbon (wing span, speed-colored)')

# Colorbar
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(speeds)
cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
cbar.set_label('Speed (m/s)')

# Axis labels and limits
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_zlim(bottom=0, top=cube_size_z)
ax.set_xlim(0, cube_size_x)
ax.set_ylim(0, cube_size_y)
ax.set_box_aspect([cube_size_x, cube_size_y, cube_size_z])

ax.set_title(f'Aircraft Path in {cube_size_x}x{cube_size_y}m Arena\n'
             f'Sphere of Awareness (SOA) Radius: {soa_radius}m')

# Legend
from matplotlib.patches import Patch
hill_patch = Patch(color='tan', label=f'{num_hills} Parabolic Hills', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
handles.append(hill_patch)
labels.append(f'{num_hills} Parabolic Hills')
ax.legend(handles, labels, loc='upper left')

plt.tight_layout()
plt.show()

# --- Evasion Statistics Calculation ---
# Check if the simulation ended prematurely in a crash
crashed = (speeds[-1] == 0) and (t < total_time - dt)

# Print total flight time
print(f"\nTotal Flight Time: {t:.2f}s")

if crashed and len(soa_activations) > 0:
    print("\n--- CRASH DETECTED: Analyzing Final Evasion Attempt ---")
    
    # Analyze final lift vs. avoidance vector alignment
    norm_lift_final = np.linalg.norm(final_lift_vec)
    norm_avoid = np.linalg.norm(final_avoidance_vec)

    # The index of the final data point just before the crash state was recorded
    crash_timestep_index = len(soa_activations) - 1
    last_activation_start_index = -1

    # Find the start of the last continuous SOA activation period before the crash.
    # First, find the very last moment SOA was active.
    last_active_index = -1
    for i in range(crash_timestep_index, -1, -1):
        if soa_activations[i]:
            last_active_index = i
            break
    
    # If we found an active SOA period leading to the crash, analyze it.
    if last_active_index != -1:
        # Trace back from the last active moment to find when that period started.
        last_activation_start_index = last_active_index
        for i in range(last_active_index - 1, -1, -1):
            if not soa_activations[i]:
                break # The period of activation has ended
            last_activation_start_index = i

        # --- Enhanced Escape Vector Alignment Calculation ---
        if norm_lift_final > 1e-6 and norm_avoid > 1e-6 and last_activation_start_index < len(lift_vectors_history):
            # 1. Calculate alignment at the moment of impact (final state)
            cos_theta_final = np.dot(normalize(final_lift_vec), normalize(final_avoidance_vec))
            final_alignment_percent = cos_theta_final * 100

            # 2. Calculate alignment at the start of the evasion maneuver
            initial_lift_vec = lift_vectors_history[last_activation_start_index]
            norm_lift_initial = np.linalg.norm(initial_lift_vec)
            
            if norm_lift_initial > 1e-6:
                cos_theta_initial = np.dot(normalize(initial_lift_vec), normalize(final_avoidance_vec))
                initial_alignment_percent = cos_theta_initial * 100
                
                # 3. Calculate the change in alignment
                alignment_change = final_alignment_percent - initial_alignment_percent
                print(f"Escape Vector Alignment: {final_alignment_percent:.2f}% ({alignment_change:+.2f}%)")
            else:
                # Fallback if initial lift was zero
                print(f"Escape Vector Alignment: {final_alignment_percent:.2f}% (Initial lift was zero)")

        else:
            print("Could not calculate final lift alignment (no active avoidance vector or lift at impact).")
        
        # 1. Calculate the "Dodge Window"
        # The number of steps from first alert to the step before impact.
        timesteps_to_react = crash_timestep_index - last_activation_start_index
        dodge_window_sec = timesteps_to_react * dt

        # 2. Get key metrics at the start of the final evasion
        terminal_soa_entry_speed = speeds[last_activation_start_index]
        
        print(f"Terminal SOA Entry Speed: {terminal_soa_entry_speed:.2f} m/s")
        print(f"Dodge Window: {dodge_window_sec:.2f}s (Time from final alert to impact)")
        
        # 3. Calculate the "Evasion Heading Deflection"
        # Get velocity vector at the start of the alert
        v_start_evasion = velocities[last_activation_start_index]
        # Get velocity vector at the start of the final timestep (just before impact)
        v_end_evasion = velocities[crash_timestep_index]
        
        norm_v_start = np.linalg.norm(v_start_evasion)
        norm_v_end = np.linalg.norm(v_end_evasion)
        
        if norm_v_start > 0 and norm_v_end > 0:
            # Calculate the angle between the two vectors
            dot_product = np.dot(normalize(v_start_evasion), normalize(v_end_evasion))
            deflection_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            deflection_deg = np.degrees(deflection_rad)
            print(f"Evasion Heading Deflection: {deflection_deg:.2f} degrees")
            
            # 4. Calculate the "Evasion Turn Rate"
            if dodge_window_sec > 0:
                evasion_turn_rate = deflection_deg / dodge_window_sec
                print(f"Average Evasion Turn Rate: {evasion_turn_rate:.2f} deg/s")
                
                if evasion_turn_rate > 0:
                    # Convert turn rate from deg/s to rad/s for the formula R = v / omega
                    evasion_turn_rate_rad = np.radians(evasion_turn_rate)
                    turn_radius = terminal_soa_entry_speed / evasion_turn_rate_rad
                    print(f"Implicit Turn Radius: {turn_radius:.2f} m (Speed / Angular Rate)")

            else:
                print("Cannot calculate turn rate for zero-duration event.")

        else:
            print("Could not calculate heading deflection (zero velocity vector found).")

    else:
        print("Crash occurred without prior SOA activation.")
else:
    print("\nSimulation completed successfully or without a detectable crash event.")