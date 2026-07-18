<h1 align="center">
Plantation Simulation (Dynamic) & Random NPC in Unreal Engine
</h1>

<p align="center">
A comprehensive Unreal Engine simulation environment featuring
Weather System, Random NPC-AI, Dynamic Objects, and Environmental Interactions.
</p>

<p align="center">
  <img src="assets/demogif.gif" width="800">
</p>

---

## Table of Contents

- [Plantation Types](#plantation-types)
- [Weather Documentation](#weather-documentation)
- [Weather Settings](#weather-settings)
  - [Light Source](#light-source)
  - [Sky Sphere](#sky-sphere)
- [Random NPC](#random-npc)
  - [Character & Animation Preparation](#character--animation-preparation)
  - [Exporting from Blender to Unreal](#exporting-from-blender-to-unreal)
  - [Importing into Unreal Engine](#importing-into-unreal-engine)
  - [Blend Shape ID](#blend-shape-id-blend_spid)
  - [Animation Blueprint](#animation-blueprint-bp_animate)
  - [Blueprint Class](#blueprint-class-bp_character)
  - [NavMesh Bounds Volume](#navmesh-bounds-volume)

---

## Plantation Types

| Random Spacing | Standard Plantation | Identical Rows |
|---|---|---|
| <img width="248" height="248" alt="image" src="https://github.com/user-attachments/assets/0e890a39-5591-4999-87cf-9fa4fe5f8ee8" />| <img width="248" height="248" alt="image" src="https://github.com/user-attachments/assets/3ef40ba7-09cf-42bb-9c33-051c8989a8bb" /> | <img width="248" height="248" alt="image" src="https://github.com/user-attachments/assets/9c157de9-6597-4fb3-b72a-c9659d537ef2" />|
| <img width="266" height="149" alt="image" src="https://github.com/user-attachments/assets/3dc5e401-bd5f-4ac8-a16f-90e14d82993c" />| <img width="266" height="149" alt="image" src="https://github.com/user-attachments/assets/02ed3709-090b-4adf-8d2b-2e162a2849b9" />| <img width="266" height="149" alt="image" src="https://github.com/user-attachments/assets/cac1b5f5-c6a7-40b4-81ba-bba954dab813" />|

---

## Weather Documentation

| Clear Afternoon | Cloudy Afternoon |
|---|---|
|<img width="413" height="225" alt="image" src="https://github.com/user-attachments/assets/ee008ea1-b3a8-4143-877f-a94bbc2da4b9" />| <img width="413" height="225" alt="image" src="https://github.com/user-attachments/assets/27d600e3-2341-4fd9-a755-faf9ea9d7abc" />|

| Clear Evening | Cloudy Evening |
|---|---|
|<img width="413" height="227" alt="image" src="https://github.com/user-attachments/assets/5eff8d49-70dc-4115-bdc4-1993691d81f6" /> | <img width="413" height="224" alt="image" src="https://github.com/user-attachments/assets/11d6dd10-2650-44cd-b9cc-3e2d19e4748a" />|

---

## Weather Settings

The two main components for configuring weather are **Light Source** and **Sky Sphere**.

### Light Source

Use the **Transform** component's **Rotation** variable to control the direction of sunlight.

- **Intensity** — Controls the brightness of the sunlight. Higher values produce brighter light; lower values dim it.
<img width="855" height="267" alt="image" src="https://github.com/user-attachments/assets/c1b811f1-be21-49ac-b82b-0ea9fdd7304f" />

- **Light Color** — Used to change the color of the sun.
<img width="844" height="761" alt="image" src="https://github.com/user-attachments/assets/c1dc0abe-c987-4b94-8181-2bff3dc2fd43" />

### Sky Sphere

Sky Sphere handles the visual appearance of the sky and clouds. Key variables to note:

| Variable | Function |
|---|---|
| **Cloud Opacity** | Increases cloud density/intensity |
| **Sun Height** | Sets the time of day (morning, noon, evening, night) |
| Sun Brightness | Optional — overall sun brightness |
| Zenith Color | Optional — color of the upper sky |
| Horizon Color | Optional — color of the horizon |
| Cloud Color | Optional — color of the clouds |

> ⚠️ **Note:** `Sun Height` does **not** change the direction of sunlight — it only affects the visual representation of the time of day.
<img width="569" height="914" alt="image" src="https://github.com/user-attachments/assets/6ee82e06-0830-456a-86be-0a18c8096e27" />

---

## Random NPC

### Character & Animation Preparation

The NPC character uses two animations:
- **Idle** — Standing still animation
- **Walk** — Walking animation
<img width="913" height="295" alt="image" src="https://github.com/user-attachments/assets/c2d4c6ca-c35d-48bb-a5dd-0ef13f1bf6c8" />

Both animations will be used to control character movement in Unreal Engine.
<img width="912" height="1105" alt="image" src="https://github.com/user-attachments/assets/ffb9f60e-b37e-4f64-b624-5fbdd65ee4a0" />
<img width="913" height="876" alt="image" src="https://github.com/user-attachments/assets/005a5fe1-361c-4cb0-937f-d9c243bcb11e" />

---

### Exporting from Blender to Unreal

Once the character and animations are ready, export the object to Unreal Engine with the following settings:
<img width="659" height="867" alt="image" src="https://github.com/user-attachments/assets/ba510d21-9302-49cc-98e7-851a9fc31188" />

**Armature Group:**
- ❌ Disable **Add Leaf Bones**
- ✅ Enable **Only Deform Bones**

**Animation Group:**
- ❌ Disable **All Actions** — to prevent unwanted animation data conversion

---

### Importing into Unreal Engine

1. Drag and drop the exported file into the `Asset` folder inside your Unreal Engine project.
2. When the import dialog appears, make sure the following are enabled:
   - ✅ **Skeletal Mesh**
   - ✅ **Import Animation**

---

### Creating the Random NPC-AI
<img width="913" height="165" alt="image" src="https://github.com/user-attachments/assets/fa33e942-aceb-484b-8afd-d87f5a51a212" />

There are **3 files** required to build the NPC-AI:

1. [Blend Shape ID](#blend-shape-id-blend_spid)
2. [Animation Blueprint](#animation-blueprint-bp_animate)
3. [Blueprint Class](#blueprint-class-bp_character)
<img width="352" height="158" alt="image" src="https://github.com/user-attachments/assets/c412ce41-02a0-4ced-8a1a-7b85241b4b6c" />

---

### Blend Shape ID (`Blend_spID`)
<img width="352" height="158" alt="image" src="https://github.com/user-attachments/assets/efd297fa-0c11-4044-8a6f-4c7430f83804" />

> Right-click in the **Content Browser** → search for `Blend Shape ID`

This file handles the animation transition between **idle** and **walk**.
<img width="562" height="368" alt="image" src="https://github.com/user-attachments/assets/aa9dd949-b62e-4e67-9d0a-7b6c295377db" />
<img width="913" height="385" alt="image" src="https://github.com/user-attachments/assets/f623e02e-28dd-42bd-ae08-f49fcf377d2e" />


**Steps:**
1. Change the **Name** and **Maximum Axis Value** variables (adjust as needed).
2. Assign the **Idle** animation to point `0.0`.
3. Assign the **Walk** animation to point `150.0`.

> 💡 These points determine how smooth the transition is from standing to walking.

---

### Animation Blueprint (`BP_Animate`)
<img width="623" height="369" alt="image" src="https://github.com/user-attachments/assets/74142bac-dd07-4bc9-ac29-54ce76e42b61" />

> Right-click in the **Content Browser** → search for `Animation Blueprint`
<img width="913" height="372" alt="image" src="https://github.com/user-attachments/assets/466392d1-9188-48aa-b7fc-264b9453a5e5" />

**In the Event Graph:**

1. Set up the required nodes.
2. Create a new variable under **My Blueprint** → **Variables**:
   - **Name:** `Speed`
   - **Variable Type:** `Float`
3. Drag the `Speed` variable into the graph → select **Set Speed** → connect it.
4. Compile and Save.
<img width="600" height="509" alt="image" src="https://github.com/user-attachments/assets/293a1fca-a1d1-4ee5-84e1-8fb0b35aa198" />

**In the AnimGraph:**
<img width="913" height="242" alt="image" src="https://github.com/user-attachments/assets/cf9d87c1-6491-492a-a83c-3fafb70f1e30" />

1. Add the `Speed` variable using **Get Speed**.
2. Connect the **Blend Shape ID** created earlier.
3. Arrange the nodes accordingly.
4. Compile and Save.
<img width="291" height="448" alt="image" src="https://github.com/user-attachments/assets/ae094e8e-98e7-4c08-92d1-20ae822c0924" />

---

### Blueprint Class (`BP_Character`)
<img width="913" height="513" alt="image" src="https://github.com/user-attachments/assets/f0c0e430-0d2a-4be8-a419-ec7513960b17" />

-click in the **Content Browser** → search for `Blueprint Class`

This is the **core component** of the Random NPC-AI.

<img width="452" height="291" alt="image" src="https://github.com/user-attachments/assets/f33127e3-ff35-400f-9f92-3f0d56740af5" />

**Character Mesh Setup:**
1. Click **Mesh (Character Mesh)** in the components panel.
2. In the details panel on the right, assign the character object to **Skeletal Mesh**.
   
<img width="777" height="545" alt="image" src="https://github.com/user-attachments/assets/f477e182-55ad-439b-ab08-733d44f9073b" />

**Character Movement Setup:**
1. Click **Character Movement** in the components panel.
2. Find `Max Walk Speed` in the details panel.
3. Set the value to `150.0` (matching the value set in Blend Shape ID).
4. Compile and Save.


**Random AI Move Setup (in Event Graph):**

<img width="913" height="475" alt="image" src="https://github.com/user-attachments/assets/21aab228-a736-4365-9088-a595424dcb24" />


1. Create the following nodes:
   - **GetRandomReachablePointInRadius** → set `Radius` to `2000`
     > The larger the radius, the wider the area the NPC will roam.
   - **Delay** → set `Duration` as needed
     > This is how long the NPC pauses after reaching its destination.
2. Connect both **On Success** and **On Fail** conditions back to the same action loop, so the NPC continuously searches for a new destination.
3. Compile and Save.

---

### NavMesh Bounds Volume

<img width="614" height="192" alt="image" src="https://github.com/user-attachments/assets/a997549d-e234-42b4-bfea-93d62d4b95b9" />

**NavMeshBoundsVolume** is used to define the area in which NPCs can navigate on the map.

<img width="913" height="398" alt="image" src="https://github.com/user-attachments/assets/aa2b1de6-af7d-433b-a11b-9f4d67ba328c" />

**How to add it:**
1. Go to **Place Actors** → search for `NavMeshBoundsVolume`.
2. Place it in the map and adjust its size to cover the desired area.

<img width="913" height="513" alt="image" src="https://github.com/user-attachments/assets/4ed91b58-1a70-47ee-bf67-4a00953fa28a" />
<img width="913" height="513" alt="image" src="https://github.com/user-attachments/assets/15432d62-a222-467a-83f7-faf4fd9f201d" />


**Color Guide:**
| Color | Description |
|---|---|
| 🟢 Green | Area accessible to NPCs |
| No color | Area **not** accessible to NPCs |

> 💡 Press **`P`** on your keyboard to toggle the green overlay on/off.

**Final Step:**

Drag and drop the **Blueprint Class (`BP_Character`)** into the map, then duplicate it as many times as needed. The NPCs will now walk around randomly and continuously. ✅
