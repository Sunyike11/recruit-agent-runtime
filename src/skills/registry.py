from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.skills.base import BaseSkill


class SkillAlreadyRegisteredError(ValueError):
    pass


class SkillNotFoundError(KeyError):
    pass


class SkillRegistry:
    """In-memory skill registry with deterministic version lookup."""

    def __init__(self):
        self._skills: Dict[Tuple[str, str], BaseSkill] = {}
        self._versions_by_name: Dict[str, List[str]] = defaultdict(list)

    def register(self, skill: BaseSkill, allow_replace: bool = False) -> BaseSkill:
        key = (skill.spec.name, skill.spec.version)
        if key in self._skills and not allow_replace:
            raise SkillAlreadyRegisteredError(
                f"Skill already registered: {skill.spec.name}@{skill.spec.version}"
            )
        if key not in self._skills:
            self._versions_by_name[skill.spec.name].append(skill.spec.version)
        self._skills[key] = skill
        return skill

    def get(self, name: str, version: Optional[str] = None) -> BaseSkill:
        selected_version = version or self._latest_version(name)
        key = (name, selected_version)
        if key not in self._skills:
            raise SkillNotFoundError(f"Skill not found: {name}@{selected_version}")
        return self._skills[key]

    def list_skills(self) -> List[BaseSkill]:
        return [
            self._skills[(name, version)]
            for name in sorted(self._versions_by_name)
            for version in self._versions_by_name[name]
        ]

    def unregister(self, name: str, version: Optional[str] = None):
        if version is None:
            if name not in self._versions_by_name:
                raise SkillNotFoundError(f"Skill not found: {name}")
            for existing_version in list(self._versions_by_name[name]):
                self._skills.pop((name, existing_version), None)
            self._versions_by_name.pop(name, None)
            return

        key = (name, version)
        if key not in self._skills:
            raise SkillNotFoundError(f"Skill not found: {name}@{version}")
        self._skills.pop(key)
        versions = self._versions_by_name[name]
        versions.remove(version)
        if not versions:
            self._versions_by_name.pop(name, None)

    def _latest_version(self, name: str) -> str:
        versions = self._versions_by_name.get(name)
        if not versions:
            raise SkillNotFoundError(f"Skill not found: {name}")
        return versions[-1]
