from agent_policy_debate import DebateAgentPolicyMixin
from agent_policy_role import RoleAgentPolicyMixin
from agent_policy_single import SingleAgentPolicyMixin
from agent_policy_voting import VotingAgentPolicyMixin


class AgentPolicyMixin(
    SingleAgentPolicyMixin,
    VotingAgentPolicyMixin,
    RoleAgentPolicyMixin,
    DebateAgentPolicyMixin,
):
    pass
