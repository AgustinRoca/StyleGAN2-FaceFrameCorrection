<template>
  <div class="text-center">
    <v-dialog v-model="show">
      <v-card style="padding: 10px">
        <img :src="'data:image/png;base64, ' + imgs_bytes[img_index]"/>
        <v-card-text>
          <v-container>
            <v-row>
              <v-col md="12" class="d-flex justify-center">
                ID: {{ ids[img_index] }}
              </v-col>
            </v-row>
            <v-row justify="center" v-if="imgs_bytes.length > 1">
              <v-col md="2" class="d-flex justify-center">
                <v-btn
                  variant="outlined"
                  icon
                  color="primary"
                  @click="prevImage"
                >
                  <v-icon>mdi-chevron-left</v-icon>
                </v-btn>
              </v-col>
              <v-col md="2" class="d-flex justify-center">
                <v-btn
                  variant="outlined"
                  icon
                  color="primary"
                  @click="nextImage"
                >
                  <v-icon>mdi-chevron-right</v-icon>
                </v-btn>
              </v-col>
            </v-row>
          </v-container>
        </v-card-text>
        <v-card-actions>
          <v-btn color="primary" @click="closeModal">Close Dialog</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
export default {
    props: ['showModal', 'imgs_bytes', 'ids'],
    data() {
        return {
            show: false,
            img_index: 0
        }
    },
    methods: {
        closeModal(){
            this.img_index = 0
            this.$emit('close')
        },
        nextImage(){
            this.img_index = (this.img_index + 1) % this.imgs_bytes.length
        },
        prevImage(){
          this.img_index = (this.img_index + this.imgs_bytes.length - 1) % this.imgs_bytes.length
        },
    },
    watch: {
        showModal(newValue, oldValue) {
            this.show = newValue
        }
    }
}
</script>

<style scoped>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 40%;
}
</style>