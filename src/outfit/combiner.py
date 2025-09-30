import torch
import clip

class OutfitCombiner:
    """Creates outfit combinations from individual items"""
    
    def __init__(self, recommender):
        self.recommender = recommender
        from .rules import OutfitRules
        self.outfit_rules = OutfitRules()
    
    def create_outfit_combinations(self, text_query, top_k=3):
        """
        Create complete outfit combinations for a text query
        
        Args:
            text_query: What you're searching for (e.g., "casual brunch")
            top_k: How many outfit combinations to return
            
        Returns:
            List of outfit combinations with scores
        """
        
        individual_scores = {}
        
        # Encode the text query
        text_tokens = clip.tokenize([text_query]).to(self.recommender.device)
        with torch.no_grad():
            text_features = self.recommender.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Score each item against the query
        for image_path, image_features in self.recommender.clothing_database.items():
            similarity = torch.cosine_similarity(
                text_features.cpu(),
                image_features,
                dim=-1
            ).item()
            individual_scores[image_path] = similarity
        
        print(f"   Scored {len(individual_scores)} individual items")
        
        all_combinations = []
        
        for template in self.outfit_rules.outfit_templates:
            if self.outfit_rules.can_create_outfit(template, self.recommender.categorized_items.keys()):
                print(f"   Trying template: {template['name']}")
                
                # Generate combinations for this template
                combinations = self._generate_combinations_for_template(
                    template, individual_scores, text_query
                )
                all_combinations.extend(combinations)
        
        all_combinations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        print(f"   Generated {len(all_combinations)} total combinations")
        print(f"   Returning top {top_k}")
        
        return all_combinations[:top_k]
    
    def _generate_combinations_for_template(self, template, individual_scores, query):
        combinations = []
        
        category_items = {}
        
        for category in template['required']:
            if category not in self.recommender.categorized_items:
                continue
            
            items = self.recommender.categorized_items[category]
            
            items_with_scores = []
            for item in items:
                score = individual_scores.get(item['path'], 0)
                items_with_scores.append((item, score))
            
            items_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            category_items[category] = items_with_scores[:3]
                
        if template['name'] == 'separates_outfit':
            combinations.extend(
                self._create_separates_combinations(category_items, individual_scores, template)
            )
        
        elif template['name'] == 'dress_outfit':
            combinations.extend(
                self._create_dress_combinations(category_items, individual_scores, template)
            )
        
        elif template['name'] == 'layered_outfit':
            combinations.extend(
                self._create_layered_combinations(category_items, individual_scores, template)
            )
        
        return combinations
    
    def _create_separates_combinations(self, category_items, individual_scores, template):
        combinations = []
        
        if 'tops' not in category_items or 'bottoms' not in category_items:
            return combinations
        
        for top_item, top_score in category_items['tops']:
            for bottom_item, bottom_score in category_items['bottoms']:
                
                # Create the outfit
                outfit = {
                    'tops': top_item,
                    'bottoms': bottom_item
                }
                
                # Add optional shoes if available
                if 'shoes' in self.recommender.categorized_items:
                    best_shoes = max(
                        self.recommender.categorized_items['shoes'],
                        key=lambda x: individual_scores.get(x['path'], 0)
                    )
                    outfit['shoes'] = best_shoes
                
                compatibility_score = self._calculate_compatibility(outfit)
                
                # Overall score: combination of individual scores + compatibility
                avg_individual_score = (top_score + bottom_score) / 2
                overall_score = (avg_individual_score * 0.6) + (compatibility_score * 0.4)
                
                combinations.append({
                    'template': template['name'],
                    'outfit': outfit,
                    'individual_scores': {
                        'tops': top_score,
                        'bottoms': bottom_score
                    },
                    'compatibility_score': compatibility_score,
                    'overall_score': overall_score
                })
        
        return combinations
    
    def _create_dress_combinations(self, category_items, individual_scores, template):
        combinations = []
        
        if 'dresses' not in category_items:
            return combinations
        
        for dress_item, dress_score in category_items['dresses']:
            outfit = {
                'dresses': dress_item
            }
            
            # Add optional shoes if available
            if 'shoes' in self.recommender.categorized_items:
                best_shoes = max(
                    self.recommender.categorized_items['shoes'],
                    key=lambda x: individual_scores.get(x['path'], 0)
                )
                outfit['shoes'] = best_shoes
            
            compatibility_score = 0.5  # Neutral
            overall_score = (dress_score * 0.8) + (compatibility_score * 0.2)
            
            combinations.append({
                'template': template['name'],
                'outfit': outfit,
                'individual_scores': {'dresses': dress_score},
                'compatibility_score': compatibility_score,
                'overall_score': overall_score
            })
        
        return combinations
    
    def _create_layered_combinations(self, category_items, individual_scores, template):
        combinations = []
        
        required = ['tops', 'bottoms', 'outerwear']
        if not all(cat in category_items for cat in required):
            return combinations
        
        # Try combining top + bottom + outerwear
        for top_item, top_score in category_items['tops']:
            for bottom_item, bottom_score in category_items['bottoms']:
                for outer_item, outer_score in category_items['outerwear']:
                    
                    outfit = {
                        'tops': top_item,
                        'bottoms': bottom_item,
                        'outerwear': outer_item
                    }
                    
                    compatibility_score = self._calculate_compatibility(outfit)
                    
                    avg_individual_score = (top_score + bottom_score + outer_score) / 3
                    overall_score = (avg_individual_score * 0.6) + (compatibility_score * 0.4)
                    
                    combinations.append({
                        'template': template['name'],
                        'outfit': outfit,
                        'individual_scores': {
                            'tops': top_score,
                            'bottoms': bottom_score,
                            'outerwear': outer_score
                        },
                        'compatibility_score': compatibility_score,
                        'overall_score': overall_score
                    })
        
        return combinations
    
    def _calculate_compatibility(self, outfit):
        outfit_items = list(outfit.values())
        
        if len(outfit_items) < 2:
            return 0.7  # Neutral score for single items
        
        # Calculate how similar the items are to each other
        total_similarity = 0
        pair_count = 0
        
        for i in range(len(outfit_items)):
            for j in range(i + 1, len(outfit_items)):
                item1_path = outfit_items[i]['path']
                item2_path = outfit_items[j]['path']
                
                item1_features = self.recommender.clothing_database[item1_path]
                item2_features = self.recommender.clothing_database[item2_path]
                
                similarity = torch.cosine_similarity(
                    item1_features,
                    item2_features,
                    dim=-1
                ).item()
                
                total_similarity += similarity
                pair_count += 1
        
        avg_compatibility = total_similarity / pair_count if pair_count > 0 else 0.5
        
        # Good range is around 0.2-0.4 similarity
        if 0.2 <= avg_compatibility <= 0.4:
            return 0.8  # Good compatibility
        elif avg_compatibility < 0.2:
            return 0.5  # Items might be too different
        else:
            return 0.6  # Items might be too similar