class OutfitRules:
    """Defines valid outfit combinations"""
    
    def __init__(self):
        self.outfit_templates = [
            {
                'name': 'separates_outfit',
                'description': 'Top and bottom combination',
                'required': ['tops', 'bottoms'],
                'optional': ['shoes', 'outerwear'],
                'min_items': 2,
                'max_items': 4
            },
            {
                'name': 'dress_outfit',
                'description': 'Dress-based outfit',
                'required': ['dresses'],
                'optional': ['shoes', 'outerwear'],
                'min_items': 1,
                'max_items': 3
            },
            {
                'name': 'layered_outfit',
                'description': 'Outfit with outerwear',
                'required': ['tops', 'bottoms', 'outerwear'],
                'optional': ['shoes'],
                'min_items': 3,
                'max_items': 4
            }
        ]
    
    def can_create_outfit(self, template, available_categories):
        """Check if we have items to create this outfit type"""
        for required_category in template['required']:
            if required_category not in available_categories:
                return False
        return True