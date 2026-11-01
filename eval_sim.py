import numpy as np
from stable_baselines3 import PPO
import genesis as gs

class EvalGenesisArmEnv:
    """Evaluation environment wrapper that attaches a camera before build()."""
    def __init__(self):
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=False)
        
        # Add Entities
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.arm = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/lss_arm.urdf", fixed=True)
        )
        
        # Add Camera BEFORE scene.build()
        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(0.8, 0.8, 0.6),
            lookat=(0.15, 0.0, 0.2),
            fov=45
        )
        
        # Build scene after camera is added
        self.scene.build()
        self.n_dofs = self.arm.n_dofs
        self.target_pos = np.array([0.2, 0.0, 0.2], dtype=np.float32)

    def reset(self):
        self.arm.set_dofs_position(np.zeros(self.n_dofs))
        self.scene.step()
        
        self.target_pos = np.array([
            np.random.uniform(0.15, 0.3),
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(0.1, 0.3)
        ], dtype=np.float32)
        
        return self._get_obs()

    def _get_obs(self):
        current_joints = self.arm.get_dofs_position().numpy()
        return np.concatenate([current_joints, self.target_pos]).astype(np.float32)

    def step(self, action):
        current_joints = self.arm.get_dofs_position().numpy()
        target_joints = current_joints + action
        
        self.arm.control_dofs_position(target_joints)
        self.scene.step()
        
        ee_pos = self.arm.get_link('lss_arm_link_4').get_pos().numpy()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        terminated = bool(distance < 0.02)
        return self._get_obs(), distance, terminated

def evaluate():
    env = EvalGenesisArmEnv()
    model = PPO.load("lss_arm_policy")
    
    print("Evaluating trained policy and recording video...")
    env.cam.start_recording()
    
    distances = []
    
    for episode in range(5):
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, distance, done = env.step(action)
            
            env.cam.render()
            step_count += 1
            
        distances.append(distance)
        print(f"Episode {episode + 1} finished in {step_count} steps. Final distance: {distance*100:.2f} cm")

    # Save recorded video
    env.cam.stop_recording(save_to_filename="trained_arm_eval.mp4", fps=30)
    print(f"\nAverage Final Distance: {np.mean(distances)*100:.2f} cm")
    print("Saved evaluation video to 'trained_arm_eval.mp4'!")

if __name__ == "__main__":
    evaluate()
