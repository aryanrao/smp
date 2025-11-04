def activity_selection(activities):
    """
    Solves the Activity Selection problem with a greedy algorithm.
    [cite: 102, 132]
    
    The greedy choice is to always pick the next activity that
    finishes earliest, among those that don't conflict with
    the last-chosen activity.

    Args:
        activities (list): A list of (start, finish) tuples.

    Returns:
        list: A list of (start, finish) tuples representing the
              maximum set of non-overlapping activities.
    """
    if not activities:
        return []

    # 1. The crucial step: sort activities by their FINISH time.
    sorted_activities = sorted(activities, key=lambda x: x[1])

    selected = []
    
    # 2. Select the first activity (which finishes earliest)
    last_activity = sorted_activities[0]
    selected.append(last_activity)
    last_finish_time = last_activity[1]

    # 3. Iterate through the rest
    for activity in sorted_activities[1:]:
        start, finish = activity
        
        # If this activity starts after the last one finished, select it
        if start >= last_finish_time:
            selected.append(activity)
            last_finish_time = finish

    return selected