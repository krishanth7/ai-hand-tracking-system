from utils import calculate_distance

class GestureRecognizer:
    """
    Analyzes hand landmarks and finger states to recognize specific gestures.
    """
    
    @staticmethod
    def recognize(hand, fingers):
        """
        Recognizes gestures based on finger states and landmark distances.
        """
        lm_list = hand["lm_list"]
        
        # 1. Open Palm: All fingers up
        if fingers == [1, 1, 1, 1, 1]:
            return "Open Palm"
            
        # 2. Fist: All fingers down
        if fingers == [0, 0, 0, 0, 0]:
            return "Fist"
            
        # 3. Pointing: Only Index up
        if fingers == [0, 1, 0, 0, 0]:
            return "Pointing"
            
        # 4. Peace: Index and Middle up
        if fingers == [0, 1, 1, 0, 0]:
            return "Peace"

        # 5. Pinch: Thumb and Index tips are very close
        # Landmark 4: Thumb Tip, Landmark 8: Index Tip
        dist = calculate_distance(lm_list[4], lm_list[8])
        if dist < 40 and fingers[1:] == [1, 0, 0, 0]:
            return "Pinch"

        return "Unknown"
