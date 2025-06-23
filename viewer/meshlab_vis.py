import subprocess

class MeshLabVis:
    def __init__(self, reconstruct_provider):
        self.reconstruct_provider = reconstruct_provider
        self.meshlab = "/home/simba/Documents/software/MeshLab2022.02-linux.AppImage"

    def show(self):
        mesh_file = self.reconstruct_provider.get_obj_model_latest_file()
        if mesh_file is None:
            print("No mesh file found")
            return
        print(f"Showing mesh file: {mesh_file}")
        cmd = f'''{self.meshlab}  "{mesh_file}"'''
        subprocess.run(cmd, shell=True)
