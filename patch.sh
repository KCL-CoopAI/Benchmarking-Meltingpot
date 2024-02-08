# This patch fixes two bugs in ray when using a conda environment.

# Find Python folder name so that this patch can run correctly on different versions of Python.
python_loc_name=$(which python)
python_version=python$(python -c "import sys; print(sys.version_info[0], sys.version_info[1], sep='.')")
python_folder_name="${python_loc_name::-10}"lib/${python_version}
echo $python_folder_name

# Apply patches from gym to gymnasium.
# sed -i '3s/gym/gymnasium/' "$python_folder_name"/site-packages/supersuit/generic_wrappers/frame_stack.py # Hardcoded observation space to uint8.
# sed -i '2s/gym/gymnasium/' "$python_folder_name"/site-packages/supersuit/utils/frame_stack.py # Hardcoded observation space to uint8.
# sed -i '182c\
#         if isinstance(self._observations, dict):\
#             return self._observations[agent]\
#         else:\
#             return self._observations[0][agent]' "$python_folder_name"/site-packages/pettingzoo/utils/conversions.py # Hardcoded observation space to uint8.
sed -i '206s/obss, rews, dones, infos = self.env.step(self._actions)/obss, rews, dones, _, infos = self.env.step(self._actions)/' "$python_folder_name"/site-packages/pettingzoo/utils/conversions.py # Hardcoded observation space to uint8.