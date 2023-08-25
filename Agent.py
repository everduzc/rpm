# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np
import cv2

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):
        if problem.problemType == "2x2":
            figures = problem.figures
            image_A = figures["A"]
            image_B = figures["B"]
            image_C = figures["C"]

            images = [image_A, image_B, image_C]

            givens = []
            for image in images:
                cur_image = self.load_image(image)
                channel_0 = cur_image[:, :, 0]
                intensity = np.cumsum(np.cumsum(channel_0, 0), 1).std()
                givens.append(intensity)

            choices = [1, 2, 3, 4, 5, 6]

            solutions = []
            for choice in choices:
                choice_image = self.load_image(figures[str(choice)])
                channel_0 = choice_image[:, :, 0]
                intensity = np.cumsum(np.cumsum(channel_0, 0), 1).std()
                solutions.append(intensity)

            solution = np.argmin(np.abs((1.0 * givens[1] * givens[2] / givens[0]) - solutions))+ 1

            return solution
        elif problem.problemType == "3x3":
            if "Problems D" in problem.problemSetName:
                return self.clustering_3x3(problem)
            elif "Problems E" in problem.problemSetName:
                return self.solve_3x3(problem)
            else:
                return self.solve_3x3(problem)
        
    def load_image(self, figure):
        img = Image.open(figure.visualFilename)
        img = np.array(img, dtype=np.uint8)
        return img

    def solve_3x3(self,problem):
        self.problemFigures = problem.figures
        answer = -1
        image_A = self.problemFigures["A"].visualFilename
        image_C = self.problemFigures["C"].visualFilename
        image_G = self.problemFigures["G"].visualFilename
        ipr_AC = self.get_ipr(image_A, image_C)
        ipr_AG = self.get_ipr(image_A, image_G)
        miss_img = int(self.find_miss(ipr_AC,image_G))
        answer = int(self.find_miss(ipr_AG,image_C))
        return answer
    
    def black_pixels(self,image):
        cv2Image = cv2.imread(image, 0)
        num_pix = (cv2Image == 0).sum()#np.sum(cv2Image == 0)
        return num_pix
    
    def find_miss(self, ipr_diff, imageG):
        min_diff = 100
        min_img = "1"
        img_set = ["1","2","3","4","5","6","7","8"]
        for img in img_set:
            img_path = self.problemFigures[img].visualFilename
            img_ipr = self.get_ipr(imageG, img_path)
            dist_ipr = abs(ipr_diff - img_ipr)
            if dist_ipr < min_diff:
                min_diff = dist_ipr
                min_img = img
        return min_img
    
    def get_ipr(self, image_1, image_2):
        image1 = cv2.imread(image_1)
        image2 = cv2.imread(image_2)
        dest_and = cv2.bitwise_and(image1, image2)
        image1_dp = self.black_pixels(image_1)
        image2_dp = self.black_pixels(image_2)
        d_and_dp = (dest_and == 0).sum()
        image1_ipr = image1_dp / d_and_dp
        image2_ipr = image2_dp / d_and_dp
        return image1_ipr - image2_ipr

    def clustering_3x3(self,problem):
        figures = problem.figures
        figs_3by3 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        ans_3by3 = ['1','2','3','4','5','6','7','8']

        ans_indices = {key: idx + 1 for idx, key in enumerate(ans_3by3)}

        dark_pixels = [self.black_pixels(figures[figure_key].visualFilename) for figure_key in figs_3by3]
        dark_pixels_sorted = sorted(dark_pixels)
        differences = [(dark_pixels_sorted[i + 1] - dark_pixels_sorted[i], i) for i in range(len(dark_pixels_sorted) - 1)]
        differences_sorted = sorted(differences, reverse=True)

        max_difference = -1
        cluster_points = []
        flag = False
        for i in range(1, len(differences_sorted)):
            if differences_sorted[i][0] > max_difference:
                max_difference = differences_sorted[i][0]
                cluster_points = [differences_sorted[i][1], differences_sorted[i - 1][1]]

        cluster_points.sort()
        clusters = [dark_pixels_sorted[:cluster_points[0] + 1]]
        for i in range(len(cluster_points)):
            if i + 1 < len(cluster_points):
                clusters.append(dark_pixels_sorted[cluster_points[i] + 1: cluster_points[i + 1] + 1])
        clusters.append(dark_pixels_sorted[cluster_points[-1] + 1:])

        chosen_cluster = next((cluster for cluster in clusters if len(cluster) > 3), None)
        if chosen_cluster:
            target_value = np.max(chosen_cluster)
            flag = True
        else:
            chosen_cluster = clusters.pop(clusters.index(min(clusters, key=len)))
            target_value = sum(chosen_cluster) / len(chosen_cluster)

        dark_pixels_ans = [self.black_pixels(figures[figure_key].visualFilename) for figure_key in ans_3by3]
        answer = int(min(ans_3by3, key=lambda x: abs(target_value - dark_pixels_ans[ans_indices[x] - 1])))

        if flag:
            answer = answer - 1
            if answer == 0:
                answer = 4

        return answer
    def ex_logop(self,image_a, image_b, operation):
        if operation == "AND":
            return np.logical_and(image_a, image_b)
        elif operation == "OR":
            return np.logical_or(image_a, image_b)
        elif operation == "XOR":
            return np.logical_xor(image_a, image_b)
        elif operation == "NOT_XOR":
            return np.bitwise_not(np.bitwise_xor(image_a, image_b))
    
    def log_ops_3x3(self, problem):
        problem_images = []
        solution_images = []
        for key in problem.figures:
            if key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                problem_images.append(problem.figures[key])
            if key in ['1', '2', '3', '4', '5', '6', '7', '8']:
                solution_images.append(problem.figures[key])
        problem_image_C = cv2.imread(problem_images[2].visualFilename, 0)
        problem_image_G = cv2.imread(problem_images[6].visualFilename, 0)
        #horizontal
        logical_A_and_B =np.logical_and(cv2.imread(problem_images[0].visualFilename, 0),cv2.imread(problem_images[1].visualFilename, 0))
        logical_G_and_H =np.logical_and(cv2.imread(problem_images[6].visualFilename, 0),cv2.imread(problem_images[7].visualFilename, 0))
        logical_A_or_B = np.logical_or(cv2.imread(problem_images[0].visualFilename,0),cv2.imread(problem_images[1].visualFilename, 0))
        logical_G_or_H = np.logical_or(cv2.imread(problem_images[6].visualFilename,0), cv2.imread(problem_images[7].visualFilename, 0))
        logical_A_xor_B = np.logical_xor(cv2.imread(problem_images[0].visualFilename, 0),cv2.imread(problem_images[1].visualFilename, 0))
        logical_G_xor_H = np.logical_xor(cv2.imread(problem_images[6].visualFilename, 0), cv2.imread(problem_images[7].visualFilename, 0))
        #vertical
        logical_A_and_D =np.logical_and(cv2.imread(problem_images[0].visualFilename, 0),cv2.imread(problem_images[3].visualFilename, 0))
        logical_C_and_F =np.logical_and(cv2.imread(problem_images[2].visualFilename, 0),cv2.imread(problem_images[5].visualFilename, 0))
        logical_A_or_D = np.logical_or(cv2.imread(problem_images[0].visualFilename,0),cv2.imread(problem_images[3].visualFilename, 0))
        logical_C_or_F = np.logical_or(cv2.imread(problem_images[2].visualFilename,0), cv2.imread(problem_images[5].visualFilename, 0))
        logical_A_xor_D = np.logical_xor(cv2.imread(problem_images[0].visualFilename, 0),cv2.imread(problem_images[3].visualFilename, 0))
        logical_C_xor_F = np.logical_xor(cv2.imread(problem_images[2].visualFilename, 0), cv2.imread(problem_images[5].visualFilename, 0))
    
        test_and = []
        x_and = np.where(logical_A_and_B == True, 255, 0)
        y_and = np.where(logical_G_and_H == True, 255, 0)
        if self.compare(x_and, problem_image_C) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(y_and, solution_image)
                test_and.append([int(solution_images[i].name), diff])
            answer = max(test_and, key=lambda x: x[1])
            return answer[0]
        
        test_or = []
        x_or = np.where(logical_A_or_B == True, 255, 0)
        y_or = np.where(logical_G_or_H == True, 255, 0)
        if self.compare(x_or, problem_image_C) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(y_or, solution_image)
                test_or.append([int(solution_images[i].name), diff])
            answer = max(test_or, key=lambda x: x[1])
            return answer[0]
        
        test_xor = []
        x_xor = np.where(logical_A_xor_B == True, 255, 0)
        y_xor = np.where(logical_G_xor_H == True, 255, 0)
        if self.compare(x_xor, problem_image_C) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(y_xor,solution_image)
                test_xor.append([int(solution_images[i].name), diff])
            answer = max(test_xor, key=lambda x: x[1])
            return answer[0]
        
        bitwise_not_bitwise_xor = []
        bitwise_not_bitwise_xor_a_b = cv2.bitwise_not(cv2.bitwise_xor(cv2.imread(problem_images[0].visualFilename, 0), cv2.imread(problem_images[1].visualFilename, 0)))
        bitwise_not_bitwise_xor_g_h = cv2.bitwise_not(cv2.bitwise_xor(cv2.imread(problem_images[6].visualFilename, 0),cv2.imread(problem_images[7].visualFilename, 0)))
        if self.compare(bitwise_not_bitwise_xor_a_b,problem_image_C) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(bitwise_not_bitwise_xor_g_h, solution_image)
                bitwise_not_bitwise_xor.append([int(solution_images[i].name), diff])
            answer = max(bitwise_not_bitwise_xor, key=lambda x: x[1])
            return answer[0]
    
        test_and = []
        x_and = np.where(logical_A_and_D == True, 255, 0)
        y_and = np.where(logical_C_and_F == True, 255, 0)
        if self.compare(x_and, problem_image_G) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(y_and, solution_image)
                test_and.append([int(solution_images[i].name), diff])
            answer = max(test_and, key=lambda x: x[1])
            return answer[0]
        
        test_or = []
        x_or = np.where(logical_A_or_D == True, 255, 0)
        y_or = np.where(logical_C_or_F == True, 255, 0)
        if self.compare(x_or, problem_image_G) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(y_or, solution_image)
                test_or.append([int(solution_images[i].name), diff])
            answer = max(test_or, key=lambda x: x[1])
            return answer[0]
        
        test_xor = []
        x_xor = np.where(logical_A_xor_D == True, 255, 0)
        y_xor = np.where(logical_C_xor_F == True, 255, 0)
        if self.compare(x_xor, problem_image_G) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(y_xor,solution_image)
                test_xor.append([int(solution_images[i].name), diff])
            answer = max(test_xor, key=lambda x: x[1])
            return answer[0]
        
        bitwise_not_bitwise_xor = []
        bitwise_not_bitwise_xor_a_d = cv2.bitwise_not(cv2.bitwise_xor(cv2.imread(problem_images[0].visualFilename, 0), cv2.imread(problem_images[3].visualFilename, 0)))
        bitwise_not_bitwise_xor_c_f = cv2.bitwise_not(cv2.bitwise_xor(cv2.imread(problem_images[3].visualFilename, 0),cv2.imread(problem_images[5].visualFilename, 0)))
        if self.compare(bitwise_not_bitwise_xor_a_d,problem_image_G) >= 77:
            for i in range(len(solution_images)):
                solution_image = cv2.imread(solution_images[i].visualFilename, 0)
                diff = self.compare(bitwise_not_bitwise_xor_c_f, solution_image)
                bitwise_not_bitwise_xor.append([int(solution_images[i].name), diff])
            answer = max(bitwise_not_bitwise_xor, key=lambda x: x[1])
            return answer[0]
        return -1
        
    def compare(self, image_1, image_2):
        total_elements = image_1.size
        matching_elements = np.sum(np.logical_and(image_1 == image_2, image_1 == 255))

        similarity = (matching_elements / total_elements) * 100
        return similarity