# ============================================
# DEPARTMENT SUPPORT DATA
# Contains contact details for each department
# ============================================

DEPARTMENT_DATA = {
    'Water Supply': {
        'officer': 'Rajesh Sharma',
        'designation': 'Area Water Maintenance Officer',
        'phone': '1916',
        'icon': '💧',
        'color': '#0ea5e9',
        'description': 'Water supply, pipeline repairs, and maintenance'
    },
    'Electricity': {
        'officer': 'Ankit Verma',
        'designation': 'Junior Electrical Officer',
        'phone': '1912',
        'icon': '⚡',
        'color': '#f59e0b',
        'description': 'Power supply, electrical repairs, and safety'
    },
    'Sanitation': {
        'officer': 'Priya Mehta',
        'designation': 'Sanitation Supervisor',
        'phone': '155303',
        'icon': '🗑️',
        'color': '#10b981',
        'description': 'Garbage collection, drain cleaning, sanitation'
    },
    'Roads': {
        'officer': 'Aman Singh',
        'designation': 'Road Maintenance Engineer',
        'phone': '1800-123-4545',
        'icon': '🛣️',
        'color': '#6366f1',
        'description': 'Road repairs, potholes, infrastructure maintenance'
    },
    'Public Services': {
        'officer': 'Neha Kapoor',
        'designation': 'Public Health Officer',
        'phone': '104',
        'icon': '🏥',
        'color': '#ec4899',
        'description': 'Public health, pensions, government services'
    }
}

# Status workflow for complaints
STATUS_WORKFLOW = {
    'Pending': {'label': 'Pending', 'color': '#f59e0b', 'icon': '⏳', 'next': 'In Review'},
    'In Review': {'label': 'In Review', 'color': '#8b5cf6', 'icon': '🔍', 'next': 'Assigned'},
    'Assigned': {'label': 'Assigned', 'color': '#0ea5e9', 'icon': '👤', 'next': 'Resolved'},
    'Resolved': {'label': 'Resolved', 'color': '#10b981', 'icon': '✅', 'next': None}
}

def get_department_info(department_name):
    """Get department details by department name"""
    return DEPARTMENT_DATA.get(department_name, {
        'officer': 'Support Team',
        'designation': 'Customer Service',
        'phone': '1800-XXX-XXXX',
        'icon': '📞',
        'color': '#64748b',
        'description': 'General support'
    })

def get_status_info(status):
    """Get status display info"""
    return STATUS_WORKFLOW.get(status, STATUS_WORKFLOW['Pending'])