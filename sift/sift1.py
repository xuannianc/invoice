import numpy as np
import cv2


class CoverDescriptor:
    def __init__(self, useSIFT=False):
        self.useSIFT = useSIFT

    def describe(self, image):
        descriptor = cv2.BRISK_create()
        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, descs)


class CoverMatcher:
    def __init__(self, descriptor, coverPaths, ratio=0.7,
                 minMatches=40, useHamming=True):
        self.descriptor = descriptor
        self.coverPaths = coverPaths
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = "BruteForce"
        if useHamming:
            self.distanceMethod += "-Hamming"

    def search(self, queryKps, queryDescs):
        results = {}
        for coverPath in self.coverPaths:
            cover = cv2.imread(coverPath)
            gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)
            score = self.match(queryKps, queryDescs, kps, descs)
            results[coverPath] = score
        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0], reverse=True)
            return results

    def match(self, kpsA, featuresA, kpsB, featuresB):
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        # 2 表示每一个特征向量找两个最匹配的两个特征向量
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > self.minMatches:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
            return float(status.sum()) / status.size
        return -1.0


cd = CoverDescriptor(useSIFT=True)
cm = CoverMatcher(cd, ["request_part11.jpg"], ratio=0.7, minMatches=40, useHamming=False)
queryImage = cv2.imread("request_part1.jpg")
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)
results = cm.search(queryKps, queryDescs)
print(results)
if len(results) == 0:
    print("I could not find a match for that cover!")
else:
    for (i, (score, coverPath)) in enumerate(results):
        print("{}. {:.2f}% : {}".format(i + 1, score * 100, coverPath))
        result = cv2.imread(coverPath)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
