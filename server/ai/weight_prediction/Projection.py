
'''
BRODY v0.1 - Projection module

'''

import numpy as np
import matplotlib.pyplot as plt
import alphashape
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import open3d as o3d


def Find_Proj_Plane(filtered_mask_list_3d, array_3d):
    area_list = []
    polygon_type_list = []
    for i in range(len(filtered_mask_list_3d)):
        # Compute the center of mass
        center_of_mass = np.mean(filtered_mask_list_3d[i], axis=0)

        # Translate center of mass point at the origin
        centered_mask = filtered_mask_list_3d[i] - center_of_mass

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_mask, rowvar=False)

        # Compute the eigenvectors from covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Get the top 2 eigenvectors (corresponding to the largest eigenvalues)
        principal_eigenvectors = sorted_eigenvectors[:, :2]

        # Compute the normal vector using the cross product of the principal eigenvectors
        normal_vector = np.cross(principal_eigenvectors[:, 0], principal_eigenvectors[:, 1])

        # Define the plane using the normal vector and a point (center_of_mass) on the plane
        d = -np.dot(normal_vector, center_of_mass)
        plane_params = np.append(normal_vector, d)

        # Project the 3D points onto the plane
        projected_points = filtered_mask_list_3d[i] - np.dot((filtered_mask_list_3d[i] - center_of_mass), normal_vector)[:, np.newaxis] * normal_vector

        # Get the 2D coordinates of the projected points
        proj_points_2D = projected_points[:, :2]

        # MinMax Scaling
        scaler = MinMaxScaler(feature_range = (0.0,0.1))
        scaler.fit(proj_points_2D)
        transformed_xy = scaler.transform(proj_points_2D)

        # Generate Alpha shape
        shapes = alphashape.alphashape(transformed_xy, alpha=100.0)

        shapes_xy = []
        if isinstance(shapes, Polygon):
            # Extract the x and y coordinates of the exterior ring
            # print("alpha shape의 형상은 Polygon 입니다.")
            polygon_type_list.append('Polygon')
            exterior = shapes.exterior
            shapes_x, shapes_y = exterior.coords.xy
            for j in range(len(shapes_x)):
                shapes_xy.append([shapes_x[j], shapes_y[j]])
            
            # Inverse scale
            if shapes_xy:
                inverse_transformed_shapes_xy = scaler.inverse_transform(shapes_xy)
            else:
                inverse_transformed_shapes_xy = []
            
            # Set boundary point
            boundary_point_list = inverse_transformed_shapes_xy

            # # Visualize 2D points  
            # proj_X = projected_points[:, 0]
            # proj_Y = projected_points[:, 1]
            
            # # Visualize 2D boundary points
            # boundary_X = inverse_transformed_shapes_xy[:,0]
            # boundary_Y = inverse_transformed_shapes_xy[:,1]

            # plt.plot(proj_X, proj_Y, 'o', color='black', markersize=6)
            # plt.plot(boundary_X, boundary_Y, 'o', color='red', markersize=4)
            # plt.gca().invert_yaxis()
            # plt.show()

            # Calculate area of object.
            polygon = Polygon(np.array(boundary_point_list))
            polygon_area = round((polygon.area)/100,2)
            # print("실제면적:",polygon_area)
            area_list.append(polygon_area)

        elif isinstance(shapes, MultiPolygon):
            # print("alpha shape의 형상은 MultiPolygon 입니다.")
            polygon_type_list.append('MultiPolygon')
            # Iterate over the polygons in the MultiPolygon
            for polygon in shapes.geoms:
                # Extract the exterior ring of the polygon
                exterior = polygon.exterior
                # Extract the x and y coordinates of the exterior ring
                shapes_x, shapes_y = exterior.coords.xy
                for j in range(len(shapes_x)):
                    shapes_xy.append([shapes_x[j], shapes_y[j]])

            if shapes_xy:
                inverse_transformed_shapes_xy = scaler.inverse_transform(shapes_xy)
            else:
                inverse_transformed_shapes_xy = []
            boundary_point_list = inverse_transformed_shapes_xy
            
            # # 2차원 투영된 점
            # proj_X = projected_points[:, 0]
            # proj_Y = projected_points[:, 1]
            
            # # 2차원 경계점
            # boundary_X = inverse_transformed_shapes_xy[:,0]
            # boundary_Y = inverse_transformed_shapes_xy[:,1]

            # plt.plot(proj_X, proj_Y, 'o', color='black', markersize=6)
            # plt.plot(boundary_X, boundary_Y, 'o', color='red', markersize=4)
            # plt.gca().invert_yaxis()
            # plt.show()

            # Calculate area of object.
            polygon = Polygon(np.array(boundary_point_list))
            polygon_area = round((polygon.area)/100,2)
            # print("실제면적:",polygon_area)
            area_list.append(polygon_area)

        elif isinstance(shapes, GeometryCollection):
            # print("alpha shape의 형상은 GeometryCollection 입니다.")
            polygon_type_list.append('GeometryCollection')
            shapes = alphashape.alphashape(transformed_xy, alpha=500.0)

            if isinstance(shapes, MultiPolygon):
                # print('geometry + multipolygon 입니다.')
                # Iterate over the polygons in the MultiPolygon
                for polygon in shapes.geoms:
                    # Extract the exterior ring of the polygon
                    exterior = polygon.exterior
                    # Extract the x and y coordinates of the exterior ring
                    shapes_x, shapes_y = exterior.coords.xy
                    for j in range(len(shapes_x)):
                        shapes_xy.append([shapes_x[j], shapes_y[j]])

                if shapes_xy:
                    inverse_transformed_shapes_xy = scaler.inverse_transform(shapes_xy)
                else:
                    inverse_transformed_shapes_xy = []
                boundary_point_list = inverse_transformed_shapes_xy
                
                # # 2차원 투영된 점
                # proj_X = projected_points[:, 0]
                # proj_Y = projected_points[:, 1]
                
                # # 2차원 경계점
                # boundary_X = inverse_transformed_shapes_xy[:,0]
                # boundary_Y = inverse_transformed_shapes_xy[:,1]

                # plt.plot(proj_X, proj_Y, 'o', color='black', markersize=6)
                # plt.plot(boundary_X, boundary_Y, 'o', color='red', markersize=4)
                # plt.gca().invert_yaxis()
                # plt.show()

                # Calculate area of object.
                polygon = Polygon(np.array(boundary_point_list))
                polygon_area = round((polygon.area)/100,2)
                # print("실제면적:",polygon_area)
                area_list.append(polygon_area)
            
            elif isinstance(shapes, GeometryCollection):
                # print('geometry + geometry 입니다.')
                points = Polygon(transformed_xy)
                hull = points.convex_hull
                hull_points = np.array(hull.exterior.coords)

                # Inverse scale
                if hull:
                    inverse_transformed_shapes_xy = scaler.inverse_transform(hull_points)
                else: 
                    inverse_transformed_shapes_xy = []

                # Set boundary point
                boundary_point_list = inverse_transformed_shapes_xy

                # # Visualize 2D points  
                # proj_X = projected_points[:, 0]
                # proj_Y = projected_points[:, 1]
                
                # # Visualize 2D boundary points
                # boundary_X = inverse_transformed_shapes_xy[:,0]
                # boundary_Y = inverse_transformed_shapes_xy[:,1]

                # plt.plot(proj_X, proj_Y, 'o', color='black', markersize=6)
                # plt.plot(boundary_X, boundary_Y, 'o', color='red', markersize=4)
                # plt.gca().invert_yaxis()
                # plt.show()

                # Calculate area of object.
                polygon = Polygon(np.array(boundary_point_list))
                polygon_area = round((polygon.area)/100,2)
                # print("실제면적:",polygon_area)
                area_list.append(polygon_area)

            else:
                print('geometry + polygon 입니다.')
                exterior = shapes.exterior
                shapes_x, shapes_y = exterior.coords.xy
                for j in range(len(shapes_x)):
                    shapes_xy.append([shapes_x[j], shapes_y[j]])
                
                # Inverse scale
                if shapes_xy:
                    inverse_transformed_shapes_xy = scaler.inverse_transform(shapes_xy)
                else:
                    inverse_transformed_shapes_xy = []
                
                # Set boundary point
                boundary_point_list = inverse_transformed_shapes_xy

                # # Visualize 2D points  
                # proj_X = projected_points[:, 0]
                # proj_Y = projected_points[:, 1]
                
                # # Visualize 2D boundary points
                # boundary_X = inverse_transformed_shapes_xy[:,0]
                # boundary_Y = inverse_transformed_shapes_xy[:,1]

                # plt.plot(proj_X, proj_Y, 'o', color='black', markersize=6)
                # plt.plot(boundary_X, boundary_Y, 'o', color='red', markersize=4)
                # plt.gca().invert_yaxis()
                # plt.show()

                # Calculate area of object.
                polygon = Polygon(np.array(boundary_point_list))
                polygon_area = round((polygon.area)/100,2)
                # print("실제면적:",polygon_area)
                area_list.append(polygon_area)

        # # Visualize 3D 
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # X = filtered_mask_list_3d[i][:, 0]
        # Y = filtered_mask_list_3d[i][:, 1]
        # Z = filtered_mask_list_3d[i][:, 2]

        # # Create a meshgrid for the plane
        # padding = 30  # Adjust this value to increase or decrease the size of the plane
        # x_range = np.linspace(np.min(X) - padding, np.max(X) + padding, num=10)
        # y_range = np.linspace(np.min(Y) - padding, np.max(Y) + padding, num=10)
        # x, y = np.meshgrid(x_range, y_range)

        # # Calculate z values for the plane
        # A = plane_params[0]
        # B = plane_params[1]
        # C = plane_params[2]
        # D = plane_params[3]
        # z = (-D - A * x - B * y) / C

        # # Plot the plane and the points
        # ax.plot_surface(x, y, z, alpha=0.5)
        # ax.scatter(X, Y, Z, c='black', marker='o')

        # # Plot the projected points
        # proj_X = projected_points[:, 0]
        # proj_Y = projected_points[:, 1]
        # proj_Z = projected_points[:, 2]
        # ax.scatter(proj_X, proj_Y, proj_Z, c='b', marker='o', label='Projected Points')

        # # Plot the boundary points
        # boundary_X = inverse_transformed_shapes_xy[:,0]
        # boundary_Y = inverse_transformed_shapes_xy[:,1]
        # boundary_Z = (-D - A * boundary_X - B * boundary_Y) / C
        # boundary_points_3D = np.column_stack((boundary_X, boundary_Y, boundary_Z))
        # ax.scatter(boundary_X, boundary_Y, boundary_Z, c='r', marker='o', s=50, label='Boundary Points')

        # # Plot the center of mass
        # ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], c='b', marker='x', s=100, label='Center of Mass')

        # # Plot the normal vector
        # normal_scale = 35  # Adjust this value to change the length of the normal vector
        # ax.quiver(center_of_mass[0], center_of_mass[1], center_of_mass[2],
        #         normal_vector[0] * normal_scale, normal_vector[1] * normal_scale, normal_vector[2] * normal_scale,
        #         color='g', label='Normal Vector')

        # ax.set_xlabel("x coordinate")
        # ax.set_ylabel("y coordinate")
        # ax.set_zlabel("z coordinate")
        # plt.suptitle(f"{i}th point cloud of mask", fontsize=16)
        # plt.gca().invert_xaxis()
        # plt.gca().invert_zaxis()

        # # Add legend to show the center of mass and normal vector
        # ax.legend()

        # plt.show()

    return  area_list, polygon_type_list