from setuptools import find_packages, setup

package_name = 'pypackage'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='josephdoingjosephthings',
    maintainer_email='josephdoingjosephthings@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	'navigation = pypackage.navigation:main',
	'timestamp_printer = pypackage.timestamp_printer:main',
	'odom_to_base_link_tf = pypackage.odom_to_base_link_tf:main',
	'Go1Meter = pypackage.Go1Meter:main',
	'test_bed = pypackage.test_bed:main',
	'occupancy_grid_util = pypackage.occupancy_grid_util:main',
	'navigate = pypackage.navigate:main',
	'scale_urdf_inertia = pypackage.scale_urdf_inertia:main',
	'precomputed_path = pypackage.precomputed_path:main',
	'path_prediction = pypackage.path_prediction:main',
	'lpf_tuner = pypackage.lpf_tuner:main',
	'wall_normal_avoider = pypackage.wall_normal_avoider:main',
        ],
    },
)
