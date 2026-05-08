import asyncio
import os
from datetime import datetime, timedelta

import urllib3
from huesdk import Hue
from utils.log import logger

from db.schemas import HueConfiguration, HueLight, HueLightGroup
from utils.auth.hue import HueAuthProvider

urllib3.disable_warnings()

_hue_auth = HueAuthProvider()


class HueBridgeClient:
  """Client for interacting with Philips Hue bridge with configuration caching."""

  def __init__(self):
    self._instance = None

  async def _get_hue_username(self) -> str:
    try:
      username = await _hue_auth.get_username()
      logger.debug("Hue Bridge username found")
      return username
    except RuntimeError:
      logger.debug("Hue Bridge username not found, retrieving via link button")
      await _hue_auth.connect()
      return await _hue_auth.get_username()

  async def _get_hue_instance(self) -> Hue:
    try:
      username = await self._get_hue_username()
      return await asyncio.to_thread(Hue, bridge_ip=os.environ["HUE_BRIDGE_IP"], username=username)
    except Exception as error:
      logger.error(error)
      return None

  async def _sync_hue_configuration(self, instance: Hue) -> HueConfiguration:
    logger.debug("Syncing Hue configuration from bridge")

    lights = await asyncio.to_thread(instance.get_lights)
    lights_list = [
      HueLight(
        id=light.id_,
        name=light.name,
        is_on=light.is_on,
        bri=light.bri if hasattr(light, "bri") else None,
        hue=light.hue if hasattr(light, "hue") else None,
        sat=light.sat if hasattr(light, "sat") else None,
      )
      for light in lights
    ]

    groups = await asyncio.to_thread(instance.get_groups)
    groups_list = [
      HueLightGroup(id=group.id_, name=group.name)
      for group in groups
    ]

    await HueConfiguration.find_all().delete()
    config = HueConfiguration(
      lights=lights_list,
      groups=groups_list,
      lastUpdated=datetime.now().date(),
    )
    await config.insert()
    logger.debug(f"Synced {len(lights_list)} lights and {len(groups_list)} groups")
    return config

  async def get_configuration(self, force: bool = False) -> HueConfiguration:
    try:
      if not self._instance:
        self._instance = await self._get_hue_instance()
      if not self._instance:
        raise Exception("Failed to connect to Hue bridge")

      if force:
        logger.debug("Forcing fresh Hue configuration sync")
        return await self._sync_hue_configuration(self._instance)

      config_doc = await HueConfiguration.find_one()
      if config_doc:
        age = datetime.now().date() - config_doc.lastUpdated
        if age < timedelta(weeks=1):
          logger.debug(f"Using cached Hue configuration ({age.days} days old)")
          return config_doc
        else:
          logger.debug(f"Cached configuration is stale ({age.days} days old), syncing")
          return await self._sync_hue_configuration(self._instance)
      else:
        logger.debug("No cached configuration found, syncing from bridge")
        return await self._sync_hue_configuration(self._instance)
    except Exception as error:
      logger.error(f"Error getting Hue configuration: {error}")
      raise

  def get_lights_formatted(self, config: HueConfiguration) -> str:
    return ", ".join([f'{light.id}:{light.name}' for light in config.lights])

  def get_groups_formatted(self, config: HueConfiguration) -> str:
    return ", ".join([f'{group.id}:{group.name}' for group in config.groups])

  async def are_lights_on(self, target_type: str, target_id: int | None) -> bool:
    if not self._instance:
      self._instance = await self._get_hue_instance()
    if not self._instance:
      raise Exception("Failed to connect to Hue bridge")

    if target_type == "LIGHT" and target_id is not None:
      light = await asyncio.to_thread(self._instance.get_light, id_=target_id)
      return light.is_on
    elif target_type == "GROUP" and target_id is not None:
      group = await asyncio.to_thread(self._instance.get_group, id_=target_id)
      return group.is_on
    else:
      lights = await asyncio.to_thread(self._instance.get_lights)
      return any(light.is_on for light in lights)

  async def control_light(self, light_id: int, turn_on: bool) -> str:
    if not self._instance:
      self._instance = await self._get_hue_instance()
    if not self._instance:
      raise Exception("Failed to connect to Hue bridge")

    light = await asyncio.to_thread(self._instance.get_light, id_=light_id)
    await asyncio.to_thread(light.on if turn_on else light.off)
    if turn_on:
      await asyncio.to_thread(light.set_brightness, 254)
    return f"{light.name} turned {'on' if turn_on else 'off'}"

  async def control_group(self, group_id: int, turn_on: bool) -> str:
    if not self._instance:
      self._instance = await self._get_hue_instance()
    if not self._instance:
      raise Exception("Failed to connect to Hue bridge")

    group = await asyncio.to_thread(self._instance.get_group, id_=group_id)
    await asyncio.to_thread(group.on if turn_on else group.off)
    if turn_on:
      await asyncio.to_thread(group.set_brightness, 254)
    return f"{group.name} lights turned {'on' if turn_on else 'off'}"
